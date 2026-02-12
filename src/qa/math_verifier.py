"""
Math verifier for checking numeric calculations in answers
"""
from typing import List, Dict, Any, Optional
import re
import numpy as np


class MathVerifier:
    """Verify mathematical calculations in answers"""

    def __init__(self, tolerance: float = 0.01):
        """
        Initialize math verifier

        Args:
            tolerance: Relative tolerance for numeric matching (default 1%)
        """
        self.tolerance = tolerance

    def verify(
        self,
        answer: str,
        evidence: List[Dict[str, Any]],
        query: str
    ) -> Dict[str, Any]:
        """
        Verify math in answer against evidence

        Args:
            answer: Generated answer
            evidence: Retrieved evidence
            query: Original query

        Returns:
            Verification result dictionary
        """
        # Extract numbers from answer
        answer_numbers = self.extract_numbers(answer)

        # Extract numbers from evidence
        evidence_numbers = []
        for ev in evidence:
            if ev['metadata']['content_type'] == 'table':
                nums = self.extract_numbers(ev['content'])
                evidence_numbers.extend(nums)

        if not answer_numbers:
            return {
                'status': 'no_numbers',
                'message': 'No numbers found in answer'
            }

        # Detect calculation type
        calc_type = self._detect_calculation_type(query, answer)

        # Verify based on calculation type
        if calc_type == 'difference':
            verified = self._verify_difference(answer_numbers, evidence_numbers)
        elif calc_type == 'ratio':
            verified = self._verify_ratio(answer_numbers, evidence_numbers)
        elif calc_type == 'percentage':
            verified = self._verify_percentage(answer_numbers, evidence_numbers)
        else:
            # Just check if numbers appear in evidence
            verified = self._verify_presence(answer_numbers, evidence_numbers)

        if verified:
            return {
                'status': 'verified',
                'message': 'Calculations verified against evidence'
            }
        else:
            return {
                'status': 'failed',
                'message': 'Calculations do not match evidence'
            }

    def extract_numbers(self, text) -> List[float]:
        """
        Extract numbers from text or list of evidence

        Args:
            text: Input text (string) or list of evidence dictionaries

        Returns:
            List of extracted numbers
        """
        # Handle list of evidence dictionaries
        if isinstance(text, list):
            all_numbers = []
            for item in text:
                if isinstance(item, dict):
                    content = item.get('content', '')
                    all_numbers.extend(self.extract_numbers(content))
                else:
                    all_numbers.extend(self.extract_numbers(str(item)))
            return all_numbers

        # Handle string input
        text_str = str(text)

        # Pattern for numbers with optional units
        pattern = r'-?\d+\.?\d*(?:[MBmb]|million|billion|thousand)?'

        matches = re.findall(pattern, text_str)
        numbers = []

        for match in matches:
            # Convert to float, handling unit suffixes
            num_str = re.sub(r'[MBmb]|million|billion|thousand', '', match)
            try:
                num = float(num_str)

                # Apply multipliers
                if 'B' in match or 'billion' in match.lower():
                    num *= 1e9
                elif 'M' in match or 'million' in match.lower():
                    num *= 1e6
                elif 'thousand' in match.lower():
                    num *= 1e3

                numbers.append(num)
            except ValueError:
                continue

        return numbers

    def _detect_calculation_type(self, query: str, answer: str) -> str:
        """Detect type of calculation"""
        query_lower = query.lower()
        answer_lower = answer.lower()

        if any(kw in query_lower for kw in ['change', 'difference', 'increase', 'decrease']):
            return 'difference'
        elif any(kw in query_lower for kw in ['ratio', 'compared to', 'per']):
            return 'ratio'
        elif any(kw in query_lower for kw in ['percentage', '%', 'percent', 'yoy']):
            return 'percentage'
        else:
            return 'lookup'

    def _verify_difference(self, answer_nums: List[float], evidence_nums: List[float]) -> bool:
        """Verify difference calculation"""
        if len(answer_nums) < 1 or len(evidence_nums) < 2:
            return False

        # Check if answer matches difference of any two evidence numbers
        for i in range(len(evidence_nums)):
            for j in range(i + 1, len(evidence_nums)):
                diff = abs(evidence_nums[i] - evidence_nums[j])
                if self._numbers_match(answer_nums[0], diff):
                    return True

        return False

    def _verify_ratio(self, answer_nums: List[float], evidence_nums: List[float]) -> bool:
        """Verify ratio calculation"""
        if len(answer_nums) < 1 or len(evidence_nums) < 2:
            return False

        # Check if answer matches ratio of any two evidence numbers
        for i in range(len(evidence_nums)):
            for j in range(len(evidence_nums)):
                if i != j and evidence_nums[j] != 0:
                    ratio = evidence_nums[i] / evidence_nums[j]
                    if self._numbers_match(answer_nums[0], ratio):
                        return True

        return False

    def _verify_percentage(self, answer_nums: List[float], evidence_nums: List[float]) -> bool:
        """Verify percentage calculation"""
        if len(answer_nums) < 1 or len(evidence_nums) < 2:
            return False

        # Check if answer matches percentage change
        for i in range(len(evidence_nums)):
            for j in range(len(evidence_nums)):
                if i != j and evidence_nums[j] != 0:
                    pct_change = ((evidence_nums[i] - evidence_nums[j]) / evidence_nums[j]) * 100
                    if self._numbers_match(answer_nums[0], pct_change):
                        return True

        return False

    def _verify_presence(self, answer_nums: List[float], evidence_nums: List[float]) -> bool:
        """Verify numbers appear in evidence"""
        for ans_num in answer_nums:
            if not any(self._numbers_match(ans_num, ev_num) for ev_num in evidence_nums):
                return False
        return True

    def _numbers_match(self, num1: float, num2: float) -> bool:
        """Check if two numbers match within tolerance"""
        if num2 == 0:
            return abs(num1) < self.tolerance
        return abs((num1 - num2) / num2) < self.tolerance
