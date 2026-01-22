import os
import cv2
import logging
import base64
import json
import time
from typing import Dict, List, Any, TypedDict
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate , PromptTemplate
from dataclasses import dataclass
from utils.sign_stamp_agent import SignStampAgent
from utils.tesseract_OCR_agent import ocr_read_document
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = ""
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Orchestrator")

# @dataclass
class ExtractionContext(TypedDict):
    """Structured context passed between LangGraph nodes."""
    image: Any                         
    ocr_texts: str     
    marker_detections: List[Dict[str, Any]] 
    agent_results: Dict[str, Any]       
    final_output: Dict[str, Any]        
@dataclass
class extraction_context:
    ocr_texts: str
    marker_detections: List[Dict[str, Any]]

def ocr_node(state: ExtractionContext) -> Dict:
    logger.info("--- Node: Targeted OCR ---")
    text=ocr_read_document(state["image"])
    return {"ocr_texts":text}

def marker_node(state: ExtractionContext) -> Dict:
    logger.info("--- Node: Authenticity Markers ---")
    checkpoint = r"C:\Users\Abhi9\OneDrive\Documents\Convolve\stampDetectionModel\checkpoint_best_ema.pth"
    agent = SignStampAgent(checkpoint)
    detections = agent.detect_markers(state["image"])
    return {"marker_detections": [d.__dict__ for d in detections]}

def synthesis_node(state: ExtractionContext) -> Dict:
    """
    Node: VLM Synthesis
    Uses Nemotron-mini to map raw agent outputs into initial field predictions.
    This generates the 'agent_results' dict for the VLM Supervisor's planning.
    """
    logger.info("--- Node: SLM Field Mapping & Confidence Scoring ---")
    vlm = ChatOllama(model="llava:7b", temperature=0)
    
    template = """
    TASK: Map unstructured agent data into field-specific values with confidence with the help of our agent's result
    INPUTS:
    - OCR DATA: {ocr_texts}
    FIELDS TO EXTRACT BY OCR DATA : dealer name, model name, horse power, asset cost
    The dealer name is the company/organization selling the tractor/asset. Usually with key words Keywords: "Pvt Ltd", "Private Limited", "Proprietor", "Agency", "Motors", "Tractors", "Dealers"
    EXAMPLES:
    "ABC Tractors Pvt. Ltd."
    "Krishna Motors and Agency"
    "Bharat Agro Industries Private Limited"
    The model name is the specific product identifier of the tractor/asset being sold. Usually with Brand + Model Number + Variant
    Examples: "Mahindra 575 DI", "John Deere 5050D", "Swaraj 855 FE"
    horse_power is The engine power rating of the tractor, measured in HP (Horse Power). Usually in 'HP' , 'hp' , 'horse power'
    Asset_Cost is the FINAL, TOTAL amount the customer pays for the tractor/asset. With Keywords like "Total", "Grand Total", "Net Amount", "Amount Payable", "Final Amount" , symbols "/-"
    CONFIDENCE RULES:
    - 0.9+: mentioned keywords found for the specific field
    - 0.5-0.8: Field with Numeric Values 
    - <0.5: Missing data or conflicting agent results or obvious wrongs (i.e. 'horse_power' with >=3 digit number, 'asset_cost' with <=5 digit number etc) or didn't understand at all.
    SIGNATURE_STAMP_AVAILABILITY_AND_BOUNDING_BOX_COORDINATES: {marker_detections} 
    FIELDS TO EXTRACT BY SIGNATURE_STAMP_AVAILABILITY_AND_BOUNDING_BOX_COORDINATES: signature, stamp
    For field 'signature' and 'stamp' rely on the confidence given by SIGNATURE_STAMP_AVAILABILITY_AND_BOUNDING_BOX_COORDINATES
    OUTPUT: Return STRICTLY  a JSON object. Like the following example
    {{
    "dealer name":{{"value": "ABC Tractors","confidence":0.75}},
    "model name":{{"value":"Swaraj 575 DI","confidence":0.48}},
    "horse power":{{"value":45,"confidence":0.96}},
    "asset cost":{{"value":1100020,"confidence":0.32}},
    "signature":{{"value":{{"present": true, "bbox": [897.0036010742188, 1178.5089111328125, 1195.417724609375, 1289.549072265625]}},"confidence":0.98}},
    "stamp":{{"value":{{"present": false, "bbox": []}},"confidence":0.18}}
    }}
    """

    prompt = PromptTemplate.from_template(template)
    chain = prompt | vlm | JsonOutputParser()
    input_data = {
    "ocr_texts": state['ocr_texts'],
    "marker_detections": state['marker_detections']
    }
    # response = chain.invoke(input_data)
    try:
        response = chain.invoke(input_data)
        # agent_results = json.loads(response.content[response.content.find("{"):response.content.rfind("}")+1])
        # print("Printing agent results...","*"*50,agent_results,"*"*50)

        new={
            "dealer name":"dealer_name",
            "model name":"model_name",
            "horse power":"horse_power",
            "asset cost":"asset_cost"
            }
        agent_results={new.get(k, k): v for k, v in response.items()}
        return {"agent_results": agent_results}
    except Exception as e:
        logger.error(f"SLM Synthesis Failed: {e}")
        fields = ["dealer_name", "model_name", "horse_power", "asset_cost", "signature", "stamp"]
        return {"agent_results": {f: {"value": None, "confidence": 0.0} for f in fields}}

def supervisor_node(state: ExtractionContext) -> Dict:
    logger.info("--- Node: VLM Master Auditor ---")
    from utils.vlm_supervisor2 import VLMSupervisor
    vlm = VLMSupervisor()

    final_json = vlm.audit_and_extract(
        image=state['image'],
        context=extraction_context(
            ocr_texts=state['ocr_texts'],
            marker_detections=state['marker_detections'],
        ), 
        agent_results=state['agent_results'] 
    )
    return {"final_output": final_json}

def build_workflow():
    workflow = StateGraph(ExtractionContext)

    # workflow.add_node("layout", layout_node)
    workflow.add_node("ocr", ocr_node)
    workflow.add_node("markers", marker_node)
    # workflow.add_node("quality", quality_node)
    workflow.add_node("synthesis", synthesis_node) 
    workflow.add_node("supervisor", supervisor_node)
    # workflow.add_node("extract_fields",extract_fields_from_agents)

    # workflow.set_entry_point("layout")
    workflow.set_entry_point("ocr")
    # workflow.add_edge("layout", "ocr")
    workflow.add_edge("ocr", "markers")
    # workflow.add_edge("markers", "quality")
    workflow.add_edge("markers", "synthesis")
    # workflow.add_edge("quality", "synthesis") 
    workflow.add_edge("synthesis", "supervisor") 
    # workflow.add_edge("supervisor", END)
    workflow.add_edge("supervisor", END) 

    return workflow.compile()

if __name__ == "__main__":
    TEST_IMAGE_PATH = r"C:\Users\Abhi9\OneDrive\Documents\Convolve\train_set\172927572_1_pg18.png"
    img = cv2.imread(TEST_IMAGE_PATH)
    
    app = build_workflow()
    
    initial_state = {
        "image": img,
        "layout_regions": [],
        "ocr_texts": [],
        "marker_detections": [],
        "quality_metrics": {},
        "agent_results": {},
        "final_output": {}
    }
    
    logger.info("Pipeline Execution Started...")
    final_state = app.invoke(initial_state)
    
    print("\n" + "="*50)
    print("FINAL AUDITED INVOICE DATA")
    print("="*50)
    print(json.dumps(final_state["final_output"], indent=4))
