"""
VLM SUPERVISOR - OPTIMIZED VERSION
Retains original structure but fixes critical issues
"""
import json
import logging
import base64
import cv2
import time
import re
import numpy as np
import threading
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger("VLMSupervisor")

@dataclass
class ExtractionContext:
    """Structured context from other agents"""
    ocr_texts: str
    marker_detections: List[Dict[str, Any]]
    # quality_metrics: Dict[str, Any]

class VLMSupervisor:
    """
    OPTIMIZED VLM Supervisor - The Intelligent Brain
    Fixes: Prompts, Latency, Parsing, Fallback
    """
    
    def __init__(self, model_name: str = "llava:7b", device: str = "cpu"):
        logger.info(f"[VLM_SUPERVISOR] Initializing with model: {model_name}")
        self.vlm = ChatOllama(
            model=model_name,
            temperature=0.0, 
            num_ctx=2048,    
            num_predict=256,  
        )
        
        self.CONFIDENCE_THRESHOLDS = {
            'high': 0.85,   
            'medium': 0.65,  
            'low': 0.4       
        }
        
        self.FIELDS = ['dealer_name', 'model_name', 'horse_power', 'asset_cost']
        self.VISUAL_FIELDS = ['signature', 'stamp']

        self.TEXT_EXTRACTION_TIMEOUT = 300 
        self.VISUAL_VERIFICATION_TIMEOUT = 300  
        
        logger.info("[VLM_SUPERVISOR] Optimized initialization complete")
    
    # ========================================================================
    # MAIN ORCHESTRATION METHOD (OPTIMIZED)
    # ========================================================================
    
    def audit_and_extract(
        self, 
        image: np.ndarray,
        context: ExtractionContext,
        agent_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Optimized orchestration with batch extraction and fallback
        """
        start_time = time.time()
        
        logger.info("\n" + "="*70)
        logger.info("[VLM_SUPERVISOR] Starting Optimized Audit & Extraction")
        logger.info("="*70)

        img_b64 = self._encode_image(image)

        extraction_plan = self._create_extraction_plan_with_fallback(agent_results)
        

        text_fields_to_extract = []
        text_fields_to_verify = []
        
        for field in self.FIELDS:
            action = extraction_plan[field]['action']
            if action == 'extract':
                text_fields_to_extract.append(field)
            elif action == 'verify':
                text_fields_to_verify.append(field)
        

        batch_results = {}
        if text_fields_to_extract or text_fields_to_verify:
            logger.info(f"[BATCH] Extracting {len(text_fields_to_extract)} fields, verifying {len(text_fields_to_verify)} fields")
            batch_results = self._extract_fields_batch(
                fields_to_extract=text_fields_to_extract,
                fields_to_verify=text_fields_to_verify,
                img_b64=img_b64,
                context=context,
                agent_results=agent_results
            )
        
        visual_results = self._validate_visual_fields_parallel(
            img_b64=img_b64,
            context=context,
            agent_results=agent_results
        )

        final_fields = self._merge_results_with_fallback(
            agent_results=agent_results,
            batch_results=batch_results,
            visual_results=visual_results,
            extraction_plan=extraction_plan
        )
        
        processing_time = time.time() - start_time
        overall_confidence = self._calculate_confident_confidence(final_fields)
        
        result = {
            'doc_id': f"invoice_{int(time.time())}",
            'fields': {
                'dealer_name': final_fields.get('dealer_name'),
                'model_name': final_fields.get('model_name'),
                'horse_power': final_fields.get('horse_power'),
                'asset_cost': final_fields.get('asset_cost'),
                'signature': final_fields.get('signature', {'present': False, 'bbox': []}),
                'stamp': final_fields.get('stamp', {'present': False, 'bbox': []})
            },
            'confidence': round(overall_confidence, 2),
            'processing_time_sec': round(processing_time, 2),
            'cost_estimate_usd': round(processing_time * 0.000083, 5),  
            'extraction_method': 'vlm_optimized'
        }
        
        logger.info(f"\n[VLM_SUPERVISOR] Extraction Complete in {processing_time:.1f}s")
        logger.info(f"  Overall Confidence: {overall_confidence:.2%}")
        logger.info(f"  Fields extracted: {sum(1 for f in self.FIELDS if final_fields.get(f) is not None)}/4")
        
        return result
    
    def _create_extraction_plan_with_fallback(self, agent_results: Dict) -> Dict[str, Dict]:
        """
        Smart planning with fallback guarantees
        """
        plan = {}
        
        for field in self.FIELDS:
            result = agent_results.get(field, {'value': None, 'confidence': 0.0})
            confidence = result['confidence']
            value = result['value']
            
            if value is None or confidence < self.CONFIDENCE_THRESHOLDS['low']:
                action = 'extract'
            elif confidence < self.CONFIDENCE_THRESHOLDS['medium']:
                action = 'verify'
            else:
                action = 'accept'

            plan[field] = {
                'action': action,
                'agent_value': value,
                'agent_confidence': confidence,
                'fallback_value': value if value else None
            }
        
        return plan

    
    def _extract_fields_batch(
        self,
        fields_to_extract: List[str],
        fields_to_verify: List[str],
        img_b64: str,
        context: ExtractionContext,
        agent_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract multiple fields in ONE VLM call with structured JSON output
        """
        try:

            prompt = self._build_batch_prompt(
                fields_to_extract=fields_to_extract,
                fields_to_verify=fields_to_verify,
                context=context,
                agent_results=agent_results
            )
            

            response = self._call_vlm_with_timeout(
                img_b64=img_b64,
                prompt=prompt,
                timeout=self.TEXT_EXTRACTION_TIMEOUT
            )
            
            if response == "TIMEOUT":
                logger.warning("[BATCH] VLM timeout - using fallback values")
                return self._create_fallback_results(fields_to_extract + fields_to_verify, agent_results)
            
            batch_results = self._parse_batch_response(response)
            
            logger.info(f"[BATCH] Successfully extracted {len(batch_results)} fields")
            return batch_results
            
        except Exception as e:
            logger.error(f"[BATCH] Extraction failed: {e}")
            return self._create_fallback_results(fields_to_extract + fields_to_verify, agent_results)
    
    def _build_batch_prompt(
        self,
        fields_to_extract: List[str],
        fields_to_verify: List[str],
        context: ExtractionContext,
        agent_results: Dict[str, Any]
    ) -> str:
        """
        Build optimized batch prompt - POSITIVE, GUIDED, STRUCTURED
        """
        ocr_texts_str = context.ocr_texts
        if isinstance(ocr_texts_str, tuple):
            ocr_texts_str = ' '.join(ocr_texts_str)  
        ocr_snippet = ocr_texts_str[:500] if ocr_texts_str else "No OCR context"

        field_guidance = []
        
        for field in fields_to_extract + fields_to_verify:
            guidance = self._get_field_guidance(field, context, agent_results.get(field, {}).get('value'))
            field_guidance.append(f"- {field.upper()}: {guidance}")
        
        field_guidance_text = "\n".join(field_guidance)
        
        prompt = f"""You are an expert document analyzer. Extract specific fields from this invoice image.

OCR CONTEXT (for reference): {ocr_snippet}

FIELD EXTRACTION GUIDANCE:
{field_guidance_text}

IMPORTANT INSTRUCTIONS:
1. BE CONFIDENT - If you see something similar to what's needed, extract it
2. For numeric fields: Extract ONLY numbers (no symbols, text, or currency)
3. For text fields: Preserve exact wording from the document
4. If a field is truly not visible, use null
5. Return ONLY valid JSON with no additional text

VERIFICATION NOTES:
{self._build_verification_notes(fields_to_verify, agent_results)}

RETURN FORMAT (JSON):
{{
  "dealer_name": "string or null",
  "model_name": "string or null", 
  "horse_power": integer or null,
  "asset_cost": integer or null
}}

Now examine the image carefully and extract the requested fields:"""
        
        return prompt
    
    def _get_field_guidance(self, field: str, context: ExtractionContext, agent_value: Any = None) -> str:
        """
        Field-specific extraction guidance (optimized)
        """
        guidance = {
            'dealer_name': """Look at TOP of invoice for company names with "Pvt Ltd", "Motors", "Tractors". Usually largest text.""",
            'model_name': """Look in MIDDLE section for product descriptions. Pattern: Brand + Model Number (e.g., "Mahindra 575 DI").""",
            'horse_power': """Look for numbers followed by "HP" or "H.P." in specifications section. Extract only the number.""",
            'asset_cost': """Look at BOTTOM for "Total", "Amount", "₹","/-". Often handwritten. Extract largest number without symbols."""
        }
        
        base = guidance.get(field, "Extract this field from the document.")

        if context.ocr_texts:

            ocr_texts_str = context.ocr_texts
            if isinstance(ocr_texts_str, tuple):
                ocr_texts_str = ' '.join(ocr_texts_str)
            
            relevant_keywords = {
                'dealer_name': ['pvt', 'ltd', 'motors', 'tractors'],
                'model_name': ['model', 'product', 'tractor'],
                'horse_power': ['hp', 'horse', 'power'],
                'asset_cost': ['total', 'amount', '₹', 'rs']
            }
            
            keywords = relevant_keywords.get(field, [])
            if any(kw in ocr_texts_str.lower() for kw in keywords):
                base += " (OCR suggests this field is present)"
        
        return base
    
    def _build_verification_notes(self, fields_to_verify: List[str], agent_results: Dict[str, Any]) -> str:
        """
        Add verification context for fields being verified
        """
        if not fields_to_verify:
            return ""
        
        notes = ["VERIFICATION REQUIRED:"]
        for field in fields_to_verify:
            agent_value = agent_results.get(field, {}).get('value')
            if agent_value:
                notes.append(f"- For {field}, agent extracted: '{agent_value}'. Verify if this is correct.")
        
        return "\n".join(notes)
    
    def _parse_batch_response(self, response: str) -> Dict[str, Any]:
        """
        Robust parsing of VLM batch response
        """
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                result = json.loads(json_str)

                cleaned_result = {}
                for field, value in result.items():
                    if field in self.FIELDS:
                        cleaned_result[field] = self._clean_field_value(field, value)
                
                return cleaned_result
        except json.JSONDecodeError:
            logger.warning(f"[PARSER] Failed to parse JSON, trying fallback parsing")
        
        return self._parse_fallback(response)
    
    def _clean_field_value(self, field: str, value: Any) -> Any:
        """
        Clean and validate field values - FIXED: Handle None and string conversion
        """
        if value is None:
            return None
        
        value_str = str(value).strip()
        if not value_str or value_str.lower() in ['null', 'none', 'na', 'n/a', 'nan']:
            return None
        
        if field == 'horse_power':
            if isinstance(value, (int, float)):
                return int(value)
            numbers = re.findall(r'\d+', value_str)
            return int(numbers[0]) if numbers else None
        
        elif field == 'asset_cost':
            clean_str = re.sub(r'[^\d]', ' ', value_str)
            numbers = re.findall(r'\d+', clean_str)
            if numbers:
                return int(max(numbers, key=lambda x: (len(x), int(x))))
            return None
        
        else:
            return value_str if value_str else None
    
    def _parse_fallback(self, response: str) -> Dict[str, Any]:
        """
        Fallback parsing when JSON fails
        """
        result = {field: None for field in self.FIELDS}
        
        field_patterns = {
            'dealer_name': r'(?:dealer|seller)[\s:]+["\']?([^"\'\n]+)["\']?',
            'model_name': r'(?:model|product)[\s:]+["\']?([^"\'\n]+)["\']?',
            'horse_power': r'(\d+)\s*(?:HP|hp|horse power)',
            'asset_cost': r'(?:total|amount)[\s:]+[₹$]?\s*([\d,]+)'
        }
        
        for field, pattern in field_patterns.items():
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                result[field] = self._clean_field_value(field, match.group(1))
        
        return result
    
    def _create_fallback_results(self, fields: List[str], agent_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create fallback results when VLM fails
        """
        results = {}
        for field in fields:
            agent_value = agent_results.get(field, {}).get('value')
            results[field] = agent_value
        
        return results
    
    def _validate_visual_fields_parallel(
        self,
        img_b64: str,
        context: ExtractionContext,
        agent_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Parallel validation of signature and stamp
        - If agent already found signature/stamp with bounding box, use that directly
        - Otherwise, use VLM to find it
        """
        results = {'signature': None, 'stamp': None}
        
        def validate_single(marker_type: str):

            agent_result = agent_results.get(marker_type, {})
            agent_value = agent_result.get('value', {})
            

            if isinstance(agent_value, dict):
                agent_present = agent_value.get('present', False)
                agent_bbox = agent_value.get('bbox')

                if agent_present and agent_bbox and len(agent_bbox) == 4:
                    logger.info(f"[VISUAL] Agent already found {marker_type} with bbox: {agent_bbox}")
                    return {
                        'present': True,
                        'bbox': agent_bbox,
                        'confidence': agent_result.get('confidence', 0.85)  
                    }

            logger.info(f"[VISUAL] Using VLM to search for {marker_type}")
            verify_prompt = self._build_visual_verification_prompt(marker_type)
            response = self._call_vlm_with_timeout(
                img_b64, 
                verify_prompt, 
                self.VISUAL_VERIFICATION_TIMEOUT
            )
            
            return self._parse_visual_response(marker_type, response)

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_sig = executor.submit(validate_single, 'signature')
            future_stamp = executor.submit(validate_single, 'stamp')
            
            try:
                timeout = 30 
                results['signature'] = future_sig.result(timeout=timeout)
                results['stamp'] = future_stamp.result(timeout=timeout)
            except FutureTimeout:
                logger.warning(f"[VISUAL] Timeout in parallel validation after {timeout}s")
                for marker_type in ['signature', 'stamp']:
                    agent_result = agent_results.get(marker_type, {})
                    agent_value = agent_result.get('value', {})
                    if isinstance(agent_value, dict):
                        results[marker_type] = {
                            'present': agent_value.get('present', False),
                            'bbox': agent_value.get('bbox'),
                            'confidence': agent_result.get('confidence', 0.3)
                        }
                    else:
                        results[marker_type] = {'present': False, 'bbox': None, 'confidence': 0.0}
            except Exception as e:
                logger.error(f"[VISUAL] Error in parallel validation: {e}")
                for marker_type in ['signature', 'stamp']:
                    agent_result = agent_results.get(marker_type, {})
                    agent_value = agent_result.get('value', {})
                    if isinstance(agent_value, dict):
                        results[marker_type] = {
                            'present': agent_value.get('present', False),
                            'bbox': agent_value.get('bbox'),
                            'confidence': agent_result.get('confidence', 0.3)
                        }
                    else:
                        results[marker_type] = {'present': False, 'bbox': None, 'confidence': 0.0}
        
        return results
    
    def _build_visual_verification_prompt(self, marker_type: str) -> str:
        """Detailed visual verification prompt"""
        location = "bottom right" if marker_type == "signature" else "bottom left or middle"
        
        return f"""Carefully examine this invoice image and look for a {marker_type}.

IMPORTANT: Only mark as present if you can clearly see it.

LOOK HERE: Check at {location} of the invoice.
CHARACTERISTICS: {self._get_marker_characteristics(marker_type)}

RESPONSE FORMAT (follow exactly):
PRESENT: yes/no
CONFIDENCE: 0-100
BBOX: [x1,y1,x2,y2]

Example for present:
PRESENT: yes
CONFIDENCE: 85
BBOX: [650,820,950,960]

Example for absent:
PRESENT: no
CONFIDENCE: 90
BBOX: []"""
    
    def _get_marker_characteristics(self, marker_type: str) -> str:
        """Get marker characteristics"""
        if marker_type == "signature":
            return "Handwritten cursive text, often near 'Signature' or 'Sign' label"
        else:  
            return "Circular or rectangular seal, often red/blue ink, contains text in circular pattern"
    
    def _parse_visual_response(self, marker_type: str, response: str) -> Dict[str, Any]:
        """Parse visual verification response"""
        response_lower = response.lower().strip()
        

        present = False
        confidence = 0.3
        bbox = None
        
        present_patterns = [
            r'present:\s*yes',
            r'present\s*yes',
            r'yes.*present',
            r'present.*yes'
        ]
        
        for pattern in present_patterns:
            if re.search(pattern, response_lower):
                present = True
                break
        

        if not present and 'yes' in response_lower and 'no' not in response_lower:
            if re.search(r'\byes\b', response_lower):
                present = True

        conf_match = re.search(r'confidence:\s*(\d+)', response_lower)
        if conf_match:
            try:
                conf_value = int(conf_match.group(1))
                confidence = min(100, max(0, conf_value)) / 100
            except:
                confidence = 0.7 if present else 0.3

        if present:
            bbox_match = re.search(r'\[([\d\.,\s]+)\]', response)
            if bbox_match:
                try:
                    bbox_str = bbox_match.group(1)
                    bbox = [float(x.strip()) for x in bbox_str.split(',')]
                    if len(bbox) != 4:
                        bbox = None
                    else:
                        bbox = [max(0.0, min(1.0, coord)) for coord in bbox]
                except:
                    bbox = None

            if bbox is None:
                confidence = max(0.5, confidence * 0.8)  
        
        return {
            'present': present,
            'bbox': bbox if present else None,
            'confidence': confidence
        }

    def _merge_results_with_fallback(
        self,
        agent_results: Dict[str, Any],
        batch_results: Dict[str, Any],
        visual_results: Dict[str, Any],
        extraction_plan: Dict[str, Dict]
    ) -> Dict[str, Any]:
        """
        Merge all results with guaranteed fallback
        """
        final_fields = {}
        

        for field in self.FIELDS:
            plan = extraction_plan[field]
            agent_value = plan['agent_value']
            batch_value = batch_results.get(field)
            
            if plan['action'] == 'accept':

                final_fields[field] = agent_value
            
            elif plan['action'] == 'verify':
                if batch_value is not None:
                    final_fields[field] = batch_value
                else:
                    final_fields[field] = agent_value  
            
            elif plan['action'] == 'extract':
                final_fields[field] = batch_value if batch_value is not None else agent_value
            

            if field in ['horse_power', 'asset_cost'] and final_fields[field]:
                try:
                    if field == 'horse_power':
                        if isinstance(final_fields[field], str):
                            numbers = re.findall(r'\d+', final_fields[field])
                            final_fields[field] = int(numbers[0]) if numbers else None
                        else:
                            final_fields[field] = int(final_fields[field])
                    elif field == 'asset_cost':
                        if isinstance(final_fields[field], str):
                            final_fields[field] = int(re.sub(r'[^\d]', '', final_fields[field]))
                        elif isinstance(final_fields[field], (int, float)):
                            final_fields[field] = int(final_fields[field])
                except (ValueError, TypeError) as e:
                    logger.warning(f"[MERGE] Failed to clean {field}: {e}")
                    final_fields[field] = None

        for field in self.VISUAL_FIELDS:
            if visual_results.get(field):
                final_fields[field] = visual_results[field]
            else:
                agent_value = agent_results.get(field, {}).get('value', {})
                if isinstance(agent_value, dict):
                    final_fields[field] = agent_value
                else:
                    final_fields[field] = {'present': False, 'bbox': None, 'confidence': 0.0}
        
        return final_fields

    
    def _call_vlm_with_timeout(self, img_b64: str, prompt: str, timeout: int = 30) -> str:
        """
        Call VLM with timeout protection
        """
        result = {"response": None, "error": None}
        
        def worker():
            try:
                message = HumanMessage(content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{img_b64}"}
                ])
                response = self.vlm.invoke([message])
                result["response"] = response.content
            except Exception as e:
                result["error"] = str(e)
        
        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(timeout)
        
        if thread.is_alive():
            logger.warning(f"[VLM] Timeout after {timeout}s")
            return "TIMEOUT"
        
        if result["error"]:
            logger.error(f"[VLM] Error: {result['error']}")
            return "ERROR"
        
        return result["response"] or ""
    
    def _encode_image(self, image: np.ndarray) -> str:
        """Encode image to base64 with compression for faster transfer"""
        h, w = image.shape[:2]
        if h > 1024 or w > 1024:
            scale = min(1024/h, 1024/w)
            new_h, new_w = int(h * scale), int(w * scale)
            image = cv2.resize(image, (new_w, new_h))
        
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
        _, buffer = cv2.imencode('.jpg', image, encode_param)
        return base64.b64encode(buffer).decode('utf-8')
    
    def _calculate_confident_confidence(self, fields: Dict[str, Any]) -> float:
        """
        Calculate realistic confidence score
        """
        confidences = []
        
        for field in self.FIELDS:
            if fields.get(field) is not None:
                if field == 'horse_power' and isinstance(fields[field], int):
                    if 20 <= fields[field] <= 120:  
                        confidences.append(0.9)
                    else:
                        confidences.append(0.6)
                elif field == 'asset_cost' and isinstance(fields[field], int):
                    if 100000 <= fields[field] <= 2000000:  
                        confidences.append(0.9)
                    else:
                        confidences.append(0.6)
                else:
                    confidences.append(0.8) 
            else:
                confidences.append(0.2) 
        
        for field in self.VISUAL_FIELDS:
            if fields.get(field, {}).get('present'):
                confidences.append(fields[field].get('confidence', 0.7))
            else:
                confidences.append(0.5)  
        
        return sum(confidences) / len(confidences) if confidences else 0.5

    
    def test_optimized_extraction(self, image_path: str, ocr_text: str, agent_results: Dict[str, Any]):
        """
        Test method to verify improvements
        """
        image = cv2.imread(image_path)
        
        # Ensure ocr_text is string, not tuple
        if isinstance(ocr_text, tuple):
            ocr_text = ' '.join(ocr_text)
        
        context = ExtractionContext(
            ocr_texts=ocr_text,
            marker_detections=[],
            quality_metrics={'blur_score': 'low', 'language': 'eng', 'is_handwritten': False}
        )
        
        print("\n" + "="*70)
        print("TESTING OPTIMIZED VLM SUPERVISOR")
        print("="*70)
        
        start_time = time.time()
        result = self.audit_and_extract(image, context, agent_results)
        elapsed = time.time() - start_time
        
        print(f"\n✅ Extraction completed in {elapsed:.1f}s")
        print(f"📊 Overall confidence: {result['confidence']:.2%}")
        
        print("\n📋 Extracted Fields:")
        for field, value in result['fields'].items():
            agent_val = agent_results.get(field, {}).get('value')
            print(f"  {field:15} → {value} (Agent: {agent_val})")
        
        return result


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Initialize optimized supervisor
    supervisor = VLMSupervisor(model_name="llava:7b")
    
    # Test image
    test_image_path = r"C:\Users\Abhi9\OneDrive\Documents\Convolve\train\172847544_1_pg23.png"
    
    # OCR text from your OCR agent - REMOVED TRAILING COMMA
    ocr_text = """
GSTIN : 29DYVPS6080L1Z5\n\n&» vsT_ M/S SRI KALABHAI\n\nOF snus eecrons a\n\nUlukinakallu,
 Opp Pearls Club, N.H. 206, KADUR-577548, Chikkamagalur Dist.\nEmail ; kbvstkadur555(@gmail.com\nSALES | SERVICE | SPARES\n\nMob : 9900347278 / 7996987278\n\nRAVESHWARA MOTORS\n\nNo. 
 07\n\nPROFORMA INVOICE / QUOTATION Date :os-073 -263s\n\nCustomer's Name\nand Address\n\nBanker's Name\nand Address\nzDFC Fie3sT BaAvte\n\nuTD \\ea dey\n\nUnit Price\n
 \nBy fondu engine\nwith Oud dW Tye Se\nAare LOW\n\ni Q\n\nDescription (Rs.) (Nos.) | Rs.\nVST SHAKTHT MT I3Q $.00 1000\n-30 Hn? TRACTOR\n\nTotal\n\nRupees (in words) — Zevon __{ocke\n
 \n* Inciusive of excise duty & Sales Tax\n\nNOTE:\n. Payment in Advance By NEFT / RTGS\n\nAccount Number : 43846308605\n\nPAN No : DYVPS6080L,\n\nIFSC Code: SBIN0040144\n\nBranch : STATE BANK OF INDIA\n\nK.M ROAD, KADUR Branch , Kadur-577548\nPlease do not pay in CASH or DEMAND DRAFT\nor CHEQUE in favour of third party.\n\nThis proforma invoice is subject to terms and\nconditions mentioned.\n\n3. Price Subjet to Change without Notice.\n\nmM odes,\n\nCustomer Signature\n\n2.\n\nin favour of M/S SRI KALABHAIRAVESHWARA MOTORS\n\nsg\n\nAuthorised Dealer\n"
"""
    
    # Mock agent results (same as before)
    agent_results = {
        'dealer_name': {'value': 'RAVESHWARA MOTORS', 'confidence': 0.95},
        'model_name': {'value': 'VST SHAKTHI MT 130', 'confidence': 0.48},
        'horse_power': {'value': '30 HP', 'confidence': 0.25},
        'asset_cost': {'value': '700000', 'confidence': 0.18},
        'signature': {'value': {'present': False, 'bbox': []}, 'confidence': 0.98},
        'stamp': {'value': {'present': False, 'bbox': []}, 'confidence': 0.18}
    }
    
    # Run test
    result = supervisor.test_optimized_extraction(test_image_path, ocr_text, agent_results)
    
    print("\n" + "="*70)
    print("FINAL RESULT:")
    print("="*70)
    print(json.dumps(result, indent=2, ensure_ascii=False))