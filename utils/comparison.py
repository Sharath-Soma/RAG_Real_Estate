import os
import re
from typing import List, Dict, Any, Tuple, Optional
from utils.project import get_project_name, PROJECT_MAPPING

class PropertyComparison:
    """Extracts and structures comparative data from compiled Project objects."""

    COMPARISON_FIELDS = {
        "Builder": ["builder"],
        "Location": ["location"],
        "Property Type": ["property type"],
        "Configurations": ["configuration"],
        "Price": ["price"],
        "Amenities": ["amenities"],
        "Possession": ["possession"],
        "RERA": ["rera"],
        "Pros": ["pros"],
        "Cons": ["cons"],
        "Recommendation": ["recommendation"],
    }

    def __init__(self, rag_engine):
        self.rag_engine = rag_engine

    def compare_properties(self, query: str) -> Dict[str, Any]:
        """
        Compare properties based on a query like "Compare Project A vs Project B".
        Returns:
            Dictionary with comparison table and supporting documents.
        """
        properties = self._extract_property_names(query)
        if len(properties) < 2:
            return {
                "success": False,
                "error": "Please specify at least 2 properties to compare. Example: 'Compare Skyline Horizon Towers vs Meridian Garden Residencies'",
            }

        comparison_data: Dict[str, Dict[str, Any]] = {}
        all_citations: List[str] = []
        canonical_properties = []

        for prop in properties:
            data, citations = self._extract_property_data(prop)
            if not data:
                continue
            
            canonical_name = get_project_name(prop)
            if canonical_name == "General Information" and citations:
                first_doc = citations[0].split(" | ")[0]
                canonical_name = get_project_name(first_doc)
            
            if canonical_name == "General Information":
                canonical_name = prop.title()
                
            comparison_data[canonical_name] = data
            all_citations.extend(citations)
            canonical_properties.append(canonical_name)

        if not comparison_data:
            return {
                "success": False,
                "error": "Could not identify the specified projects in the database. Please check project spelling.",
            }

        comparison_table = self._build_comparison_table(comparison_data)

        return {
            "success": True,
            "properties": canonical_properties,
            "comparison_table": comparison_table,
            "citations": list(set(all_citations)),
        }

    def _extract_property_names(self, query: str) -> List[str]:
        """Extract property names from comparison query."""
        separators = [r" vs\. ", r" vs ", r" compared to ", r" versus "]
        text = query.lower()
        
        for sep in separators:
            if re.search(sep, query, re.IGNORECASE):
                parts = re.split(sep, query, flags=re.IGNORECASE)
                cleaned = []
                for part in parts:
                    part = part.strip()
                    if part.lower().startswith("compare "):
                        part = part[8:].strip()
                    if part:
                        cleaned.append(part)
                return cleaned

        if "compare" in text and " and " in text:
            compare_idx = text.find("compare")
            remainder = query[compare_idx + len("compare"):].strip()
            if remainder.lower().startswith("d "):
                remainder = remainder[2:].strip()
            parts = remainder.split(" and ")
            cleaned = [part.strip(" ?.") for part in parts if part.strip()]
            if len(cleaned) >= 2:
                return cleaned[:2]

        quoted = re.findall(r'"([^"]+)"', query)
        if len(quoted) >= 2:
            return quoted[:2]

        # Regex fallback for names matching standard keywords
        # E.g. Compare Skyline Horizon Towers and Meridian Garden Residencies
        # Let's search for matches of any known project names
        matched_canonicals = []
        for code, name in PROJECT_MAPPING.items():
            if code in text or name.lower() in text:
                matched_canonicals.append(name)
        if len(matched_canonicals) >= 2:
            return matched_canonicals[:2]

        return []

    def _extract_property_data(self, property_name: str) -> Tuple[Dict[str, Any], List[str]]:
        """Retrieve and extract structured data for a single property using compiled Project metadata."""
        canonical_name = get_project_name(property_name)
        if canonical_name == "General Information":
            # Try substring matching
            for name in self.rag_engine.projects.keys():
                if property_name.lower() in name.lower():
                    canonical_name = name
                    break

        proj = self.rag_engine.projects.get(canonical_name)
        if not proj:
            return {}, []

        details = proj.to_dict()
        
        property_data = {
            "Builder": details["Builder"],
            "Location": details["Location"],
            "Property Type": details["Property Type"],
            "Configurations": details["Configurations"],
            "Price": details["Price Range"],
            "Amenities": details["Amenities"],
            "Possession": details["Possession"],
            "RERA": details["RERA"],
        }
        
        # Dynamically generate Pros, Cons, and Verdicts based on project attributes
        model = self.rag_engine._get_gemini_model() if hasattr(self.rag_engine, '_get_gemini_model') else None
        if model:
            prompt = f"""
            Analyze the following real estate property strictly based on this extracted data:
            {property_data}
            
            Return exactly three lines with no other text:
            Pros: [1-2 concise points based strictly on facts]
            Cons: [1-2 concise points based strictly on facts, e.g., price or possession date]
            Recommendation: [Concise final verdict]
            """
            try:
                response = model.generate_content(prompt)
                text = response.text
                for line in text.split("\n"):
                    if line.startswith("Pros:"):
                        property_data["Pros"] = line.replace("Pros:", "").strip()
                    elif line.startswith("Cons:"):
                        property_data["Cons"] = line.replace("Cons:", "").strip()
                    elif line.startswith("Recommendation:"):
                        property_data["Recommendation"] = line.replace("Recommendation:", "").strip()
            except Exception as e:
                import logging
                logging.error(f"Failed to generate comparison analysis: {e}")
        
        # Fallback if Gemini fails or is missing
        if "Pros" not in property_data:
            property_data["Pros"] = "Data available (AI synthesis offline)."
            property_data["Cons"] = "Data available (AI synthesis offline)."
            property_data["Recommendation"] = "Compare factors manually."

        citations = []
        for doc in proj.documents:
            if "brochure" in doc.lower() or "payment" in doc.lower() or "rera" in doc.lower():
                citations.append(f"{doc} | page 1")

        return property_data, citations

    def _build_comparison_table(self, data: Dict[str, Dict[str, Any]]) -> List[Dict[str, str]]:
        """Build a structured comparison table with ordered standard fields."""
        all_fields = set()
        for prop in data.keys():
            # Ensure every field in COMPARISON_FIELDS exists in data[prop]
            for field in self.COMPARISON_FIELDS.keys():
                if field not in data[prop]:
                    data[prop][field] = "—"

        for prop_data in data.values():
            all_fields.update(prop_data.keys())

        # Ordered fields as explicitly requested in Section 6
        standard_fields = [
            "Builder",
            "Location",
            "Property Type",
            "Configurations",
            "Price",
            "Amenities",
            "Possession",
            "RERA",
            "Pros",
            "Cons",
            "Recommendation",
        ]
        ordered_fields = [f for f in standard_fields if f in all_fields]
        table = []
        for field in ordered_fields:
            if field.endswith("_source"):
                continue
            values = [data[prop].get(field, "—") for prop in data.keys()]
            unique_values = {value for value in values if value != "—"}
            differs = len(unique_values) > 1
            row = {"Field": field, "_differs": differs}
            for prop in data.keys():
                row[prop] = data[prop].get(field, "—")
            table.append(row)

        return table
