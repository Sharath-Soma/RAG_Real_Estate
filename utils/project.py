import os
import re
from typing import List, Dict, Any, Optional

PROJECT_MAPPING = {
    "hbp": "Horizon Blue Park",
    "mgr": "Meridian Garden Residencies",
    "mlv": "Meridian Lake View",
    "sht": "Skyline Horizon Towers",
    "unh": "Urban Nest Heights",
    "unr": "Urban Nest Residences",
}

# Maps filename prefixes/patterns that don't use the standard project code prefix
# to their canonical project name(s).  When a file belongs to multiple projects
# (e.g. a shared builder profile), list ALL canonical names.
_EXTRA_FILENAME_MAPPING: Dict[str, List[str]] = {
    "skyline_builder_profile":   ["Skyline Horizon Towers"],
    "skyline_about":             ["Skyline Horizon Towers"],
    "skyline_home":              ["Skyline Horizon Towers"],
    "skyline_faq":               ["Skyline Horizon Towers"],
    "skyline_customer_support":  ["Skyline Horizon Towers"],
    "skyline_privacy_policy":    ["Skyline Horizon Towers"],
    "skyline_terms_conditions":  ["Skyline Horizon Towers"],
    # Meridian builder profile applies to BOTH Meridian projects
    "meridian_builder_profile":  ["Meridian Garden Residencies", "Meridian Lake View"],
    "meridian_about":            ["Meridian Garden Residencies", "Meridian Lake View"],
    "meridian_home":             ["Meridian Garden Residencies", "Meridian Lake View"],
    "meridian_faq":              ["Meridian Garden Residencies", "Meridian Lake View"],
    "meridian_customer_support": ["Meridian Garden Residencies", "Meridian Lake View"],
    "meridian_privacy_policy":   ["Meridian Garden Residencies", "Meridian Lake View"],
    "meridian_terms_conditions": ["Meridian Garden Residencies", "Meridian Lake View"],
    # Urban Nest builder profile applies to BOTH Urban Nest projects
    "urbannest_builder_profile": ["Urban Nest Heights", "Urban Nest Residences"],
    "urbannest_about":           ["Urban Nest Heights", "Urban Nest Residences"],
    "urbannest_home":            ["Urban Nest Heights", "Urban Nest Residences"],
    "urbannest_faq":             ["Urban Nest Heights", "Urban Nest Residences"],
    "urbannest_customer_support":["Urban Nest Heights", "Urban Nest Residences"],
    "urbannest_privacy_policy":  ["Urban Nest Heights", "Urban Nest Residences"],
    "urbannest_terms_conditions":["Urban Nest Heights", "Urban Nest Residences"],
}


def get_project_name(filename: str) -> str:
    """Return the primary canonical project name for a given filename.

    For files shared across multiple projects (e.g. a group builder profile)
    the FIRST entry of the mapping is returned so that a single string is
    always produced.  Use ``get_all_project_names()`` when you need the full
    list for multi-project files.
    """
    base_name = os.path.basename(filename).lower()
    # Remove extension for prefix lookups
    stem = base_name.rsplit(".", 1)[0]

    # 1. Check standard project-code prefix (hbp_, mgr_, mlv_, sht_, unh_, unr_)
    for prefix, full_name in PROJECT_MAPPING.items():
        if base_name.startswith(prefix + "_") or base_name.startswith(prefix + "."):
            return full_name

    # 2. Check extra filename stems (shared / group-level files)
    for pattern, names in _EXTRA_FILENAME_MAPPING.items():
        if stem.startswith(pattern) or stem == pattern:
            return names[0]

    # 3. Keyword fallbacks (legacy / catch-all)
    if "skyline" in base_name:
        return "Skyline Horizon Towers"
    if "urbannest" in base_name or "urban_nest" in base_name:
        return "Urban Nest Heights"
    if "meridian" in base_name:
        return "Meridian Garden Residencies"

    return "General Information"


def get_all_project_names(filename: str) -> List[str]:
    """Return ALL canonical project names for a given filename.

    Most files belong to exactly one project; builder profile / group files
    belong to several and will return multiple names.
    """
    base_name = os.path.basename(filename).lower()
    stem = base_name.rsplit(".", 1)[0]

    # 1. Standard code prefix
    for prefix, full_name in PROJECT_MAPPING.items():
        if base_name.startswith(prefix + "_") or base_name.startswith(prefix + "."):
            return [full_name]

    # 2. Extra mapping (may be multi-project)
    for pattern, names in _EXTRA_FILENAME_MAPPING.items():
        if stem.startswith(pattern) or stem == pattern:
            return list(names)

    # 3. Keyword fallbacks
    if "skyline" in base_name:
        return ["Skyline Horizon Towers"]
    if "urbannest" in base_name or "urban_nest" in base_name:
        return ["Urban Nest Heights", "Urban Nest Residences"]
    if "meridian" in base_name:
        return ["Meridian Garden Residencies", "Meridian Lake View"]

    return ["General Information"]


class Project:
    """Represents a unified real estate project with structured metadata and document attachments."""

    def __init__(self, name: str):
        self.name = name
        self.builder = "Unknown"
        self.location = "Unknown"
        self.city = "Unknown"
        self.property_type = "Residential"
        self.configuration = "2/3 BHK Apartments"
        self.price_range = "Unknown"
        self.amenities = "Unknown"
        self.possession = "Unknown"
        self.payment_plan = "Unknown"
        self.cancellation_policy = "Unknown"
        self.description = "A premium real estate project offering modern living spaces."
        self.rera = "Unknown"
        self.documents: List[str] = []
        self.citations: List[str] = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Project Name": self.name,
            "Builder": self.builder,
            "Location": self.location,
            "City": self.city,
            "Property Type": self.property_type,
            "Configurations": self.configuration,
            "Price Range": self.price_range,
            "Amenities": self.amenities,
            "Possession": self.possession,
            "Payment Plan": self.payment_plan,
            "Cancellation Policy": self.cancellation_policy,
            "Short Description": self.description,
            "RERA": self.rera,
            "_documents": self.documents,
            "_citations": self.citations,
        }


def compile_projects(metadata_chunks: List[dict]) -> Dict[str, Project]:
    """
    Parse every project and build structured Project objects.
    Ensures that metadata comes strictly from the correct document types/sections.
    Multi-project files (e.g. meridian_builder_profile) are distributed to every
    project they belong to.
    """
    # 1. Initialize Project objects for known canonical names
    projects: Dict[str, Project] = {
        name: Project(name) for name in PROJECT_MAPPING.values()
    }

    # Group chunks by project and document type
    project_doc_chunks: Dict[str, Dict[str, List[str]]] = {}
    for name in projects.keys():
        project_doc_chunks[name] = {
            "brochure": [],
            "builder_profile": [],
            "payment_plan": [],
            "cancellation": [],
            "rera": [],
            "possession": [],
            "location": [],
            "amenities": [],
            "floor_plans": [],
            "other": [],
        }

    for chunk in metadata_chunks:
        doc_name = chunk.get("document_name", "")
        # A file may belong to multiple projects (e.g. shared builder profiles)
        proj_names = get_all_project_names(doc_name)

        doc_lower = doc_name.lower()
        content = chunk.get("content", "").strip()

        # Categorize chunk by filename pattern
        if "brochure" in doc_lower:
            category = "brochure"
        elif "builder_profile" in doc_lower:
            category = "builder_profile"
        elif "payment" in doc_lower:
            category = "payment_plan"
        elif "cancellation" in doc_lower or "refund" in doc_lower:
            category = "cancellation"
        elif "rera" in doc_lower:
            category = "rera"
        elif "possession" in doc_lower:
            category = "possession"
        elif "location" in doc_lower:
            category = "location"
        elif "amenit" in doc_lower:
            category = "amenities"
        elif "floor" in doc_lower:
            category = "floor_plans"
        else:
            category = "other"

        for proj_name in proj_names:
            if proj_name not in projects:
                continue
            # Attach doc name to project (once)
            if doc_name not in projects[proj_name].documents:
                projects[proj_name].documents.append(doc_name)
            if not content:
                continue
            project_doc_chunks[proj_name][category].append(content)

    # 2. Extract metadata specifically from the correct document categories
    for name, categories in project_doc_chunks.items():
        proj = projects[name]

        # Extract Builder Name
        builder_sources = categories["builder_profile"] + categories["brochure"]
        proj.builder = _extract_field(
            builder_sources,
            ["company overview"],
            r"Company Overview\s+([A-Za-z0-9\s]+?)\s+was founded"
        )
        if not proj.builder or proj.builder == "Unknown":
            proj.builder = _extract_field(
                builder_sources,
                ["developed by"],
                r"developed by\s+([A-Za-z0-9\s]+?),?\s+located"
            )

        # Extract Location & City (only from brochure, location, or listing)
        location_sources = categories["location"] + categories["brochure"] + categories["other"]
        location_val = _extract_field(
            location_sources,
            ["located in", "location", "address", "at "],
            r"(?:located in|located at|address|location)\s*(?:is|:)?\s*([A-Za-z0-9\s,]{4,60})",
        )
        if location_val and location_val != "Unknown":
            proj.location = location_val
            if "bengaluru" in location_val.lower() or "bangalore" in location_val.lower():
                proj.city = "Bengaluru"
            elif "pune" in location_val.lower():
                proj.city = "Pune"
            elif "hyderabad" in location_val.lower():
                proj.city = "Hyderabad"

        # Extract Property Type & Configurations
        if "business" in name.lower() or ("park" in name.lower() and "Blue" in name):
            proj.property_type = "Commercial"
            proj.configuration = "Office Spaces"
        elif "villa" in name.lower():
            proj.property_type = "Residential"
            proj.configuration = "4 BHK Villas"
        else:
            proj.property_type = "Residential"
            proj.configuration = "2/3 BHK Apartments"
            if "Garden Residencies" in name or "Greens Residency" in name:
                proj.configuration = "1/2/3 BHK Apartments"

        # Extract Price Range (only from brochure)
        price_sources = categories["brochure"]
        proj.price_range = _extract_field(
            price_sources,
            ["pricing", "ranges from", "price"],
            r"(?:pricing|price|ranges from|range)\s*(?:is|:)?\s*(?:inr|rs)?\s*([0-9.]+\s*(?:lakh|crore|cr|l|c)\s*(?:-\s*[0-9.]+\s*(?:lakh|crore|cr|l|c))?)",
        )

        # Extract Amenities (from amenities guide or brochure)
        amenities_sources = categories["amenities"] + categories["brochure"]
        if amenities_sources:
            match = re.search(r"Amenities(?: Guide)?\s*[-:]?\s*(?:Recreational Amenities)?\s*[-:]?\s*(.*?)\.", amenities_sources[0], re.IGNORECASE | re.DOTALL)
            if match:
                raw_amenities = _clean_text(match.group(1).replace("\n", " "))
            else:
                raw_amenities = _clean_text(amenities_sources[0][:150].replace("\n", " "))
            
            items = re.split(r",\s*and\s+|,\s*|\s+and\s+", raw_amenities)
            clean_items = [i.strip().capitalize() for i in items if i.strip()]
            if clean_items:
                proj.amenities = "\n" + "\n".join(f"• {item}" for item in clean_items)
            else:
                proj.amenities = raw_amenities

        # Extract Possession Date (from possession doc or brochure)
        poss_sources = categories["possession"] + categories["brochure"]
        proj.possession = _extract_field(
            poss_sources,
            ["scheduled for", "possession date", "proposed date"],
            r"(?:scheduled for|date of possession|possession is|proposed date of possession)\s*(?:is|:)?\s*([A-Za-z]+\s*\d{4})",
        )

        # Extract RERA (from rera doc or brochure)
        rera_sources = categories["rera"] + categories["brochure"]
        proj.rera = _extract_field(
            rera_sources,
            ["rera"],
            r"(PRM/KA/RERA/[A-Z0-9/]+|P\d{11})"
        )

        # Extract Payment Plan (only from payment plan files)
        pay_sources = categories["payment_plan"]
        if pay_sources:
            proj.payment_plan = _clean_text(pay_sources[0].split("\n")[0])
            if len(proj.payment_plan) > 150:
                proj.payment_plan = proj.payment_plan[:150] + "..."

        # Extract Cancellation Policy (only from cancellation files)
        cancel_sources = categories["cancellation"]
        if cancel_sources:
            lines = cancel_sources[0].split("\n")
            for line in lines:
                if "forfeit" in line.lower() or "refund" in line.lower():
                    proj.cancellation_policy = _clean_text(line)
                    break
            if proj.cancellation_policy == "Unknown" and lines:
                proj.cancellation_policy = _clean_text(lines[0])
            if len(proj.cancellation_policy) > 150:
                proj.cancellation_policy = proj.cancellation_policy[:150] + "..."

        # Extract Short Description (from brochure)
        desc_sources = categories["brochure"]
        if desc_sources:
            match = re.search(r"(The project spans.*?units\.)", desc_sources[0], re.IGNORECASE | re.DOTALL)
            if match:
                proj.description = _clean_text(match.group(1).replace("\n", " "))
            else:
                proj.description = _clean_text(desc_sources[0][:200].replace("\n", " "))

    return projects


def _clean_text(text: str) -> str:
    """Helper method to remove document titles and headers from extracted text."""
    pattern = r"(?i)\b(?:Project Overview|Amenities Guide|Location Guide|Brochure|Company Profile|Document titles?)\b[-:]?\s*"
    text = re.sub(pattern, "", text)
    pattern_files = r"(?i)\b[a-z]+_(?:brochure|profile|guide|plan|summary)(?:\.pdf|\.md)?\b[-:]?\s*"
    text = re.sub(pattern_files, "", text)
    return text.strip()


def _extract_field(sources: List[str], keywords: List[str], regex_pattern: str) -> str:
    """Helper method to extract fields from correct source text."""
    for text in sources:
        lines = text.split("\n")
        for line in lines:
            for kw in keywords:
                if kw.lower() in line.lower():
                    # Attempt regex match first
                    match = re.search(regex_pattern, line, re.IGNORECASE)
                    if match:
                        return _clean_text(match.group(1))
                    # Fallback to colon split
                    if ":" in line:
                        val = line.split(":", 1)[1].strip()
                        if 3 < len(val) < 150:
                            return _clean_text(val)
                    else:
                        val = line.strip()
                        if 3 < len(val) < 150:
                            return _clean_text(val)
    return "Unknown"
