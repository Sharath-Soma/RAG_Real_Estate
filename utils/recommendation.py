import os
import re
from typing import List, Dict, Any, Tuple, Optional
from utils.project import get_project_name, PROJECT_MAPPING

class PropertyRecommendation:
    """Analyzes user preferences and recommends suitable projects using structured metadata."""

    def __init__(self, rag_engine):
        self.rag_engine = rag_engine

    def recommend_properties(self, user_query: str, conversation_history: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Recommend properties based on user preferences and constraints.
        Uses structured project metadata for scoring and ranking.
        """
        # Parse user preferences from current query and conversation history
        preferences = self._parse_preferences(user_query, conversation_history or [])

        # If preferences are completely empty, check if they asked a generic recommendation query
        if not preferences and any(w in user_query.lower() for w in ["recommend", "suggest", "best", "suitable", "option"]):
            # Set default preferences to scan all
            preferences = {"generic": True}

        if not preferences:
            return {
                "success": False,
                "error": "Please specify your preferences (e.g., BHK, budget, location, amenities). Example: 'Recommend a 3 BHK in Whitefield with a pool'",
            }

        # Score and rank all projects based on preferences
        ranked_recommendations = self._score_projects(preferences)

        return {
            "success": True,
            "preferences": preferences,
            "recommendations": ranked_recommendations,
            "match_count": len(ranked_recommendations),
        }

    def _parse_preferences(self, query: str, conversation_history: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extract structured preferences from query and conversation history."""
        prefs: Dict[str, Any] = {}
        query_lower = query.lower()
        
        # Merge budget/BHK from conversation history
        if conversation_history:
            for msg in conversation_history[-4:]:
                if msg.get("role") == "user":
                    msg_lower = msg.get("content", "").lower()
                    if "budget" in msg_lower:
                        budget_match = re.search(r"(\d+)\s*(lakh|crore|cr)", msg_lower)
                        if budget_match:
                            amount = float(budget_match.group(1))
                            if budget_match.group(2) in ("crore", "cr"):
                                amount *= 100
                            prefs["budget"] = amount
                    if "bhk" in msg_lower:
                        bhk_match = re.search(r"(\d)\s*bhk", msg_lower)
                        if bhk_match:
                            prefs["bhk"] = int(bhk_match.group(1))

        # Current query budget extraction (INR Lakhs)
        budget_match = re.search(r"(?:rs\.?|₹)?\s*([\d.]+)\s*(lakh|crore|cr|l|c)", query_lower)
        if budget_match:
            amount = float(budget_match.group(1))
            unit = budget_match.group(2)
            if unit in ("crore", "cr", "c"):
                amount *= 100
            prefs["budget"] = amount

        # BHK extraction
        bhk_match = re.search(r"(\d)\s*bhk", query_lower)
        if bhk_match:
            prefs["bhk"] = int(bhk_match.group(1))

        # Property type / villa extraction
        if "villa" in query_lower:
            prefs["property_type"] = "villa"
        elif "office" in query_lower or "commercial" in query_lower or "retail" in query_lower:
            prefs["property_type"] = "commercial"
        elif "apartment" in query_lower or "flat" in query_lower or "residential" in query_lower:
            prefs["property_type"] = "residential"

        # Amenities
        amenities = []
        amenity_keywords = {
            "swimming pool": "swimming pool",
            "pool": "swimming pool",
            "gym": "gym",
            "clubhouse": "clubhouse",
            "garden": "garden",
            "park": "park",
            "power backup": "power backup",
            "security": "security",
        }
        for keyword, amenity in amenity_keywords.items():
            if keyword in query_lower:
                amenities.append(amenity)
        if amenities:
            prefs["amenities"] = list(set(amenities))

        # Location keywords
        locations = []
        location_keywords = [
            "whitefield", "outer ring road", "hinjewadi", "baner",
            "gachibowli", "kokapet", "bengaluru", "pune", "hyderabad",
        ]
        for keyword in location_keywords:
            if keyword in query_lower:
                locations.append(keyword)
        if locations:
            prefs["locations"] = locations

        # Proximity / neighbourhood keywords (treat as location hints)
        proximity_phrases = [
            "near school", "near schools", "near metro", "near hospital",
            "near park", "near office", "near it hub", "near airport",
            "school nearby", "metro nearby",
        ]
        for phrase in proximity_phrases:
            if phrase in query_lower:
                prefs.setdefault("proximity", []).append(phrase)

        # Builder keywords
        if "skyline" in query_lower:
            prefs["builder"] = "skyline"
        elif "meridian" in query_lower:
            prefs["builder"] = "meridian"
        elif "urban nest" in query_lower:
            prefs["builder"] = "urban nest"

        # Possession preference
        if any(w in query_lower for w in ["early", "immediate", "ready", "soon", "2026"]):
            prefs["possession"] = "early"

        # Investment intent
        if any(w in query_lower for w in ["invest", "investment", "roi", "return", "rental yield", "appreciation"]):
            prefs["investment"] = True

        return prefs

    def _score_projects(self, preferences: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Score each project based on preferences using structured Project objects."""
        scored_list = []

        # Iterate over all compiled projects
        for name, proj in self.rag_engine.projects.items():
            details = proj.to_dict()
            score = 60.0  # Base score
            reasons = []
            matching_features = []

            # 1. Property Type Match (Weight: 15)
            if "property_type" in preferences:
                requested_type = preferences["property_type"]
                proj_type_lower = details["Property Type"].lower()
                proj_config_lower = details["Configurations"].lower()
                
                if requested_type == "villa" and "villa" in proj_config_lower:
                    score += 15
                    reasons.append("Premium Villa project matching your villa preference")
                    matching_features.append("Property Type: Villa")
                elif requested_type == "commercial" and "commercial" in proj_type_lower:
                    score += 15
                    reasons.append("Prime Commercial office spaces matching your office space preference")
                    matching_features.append("Property Type: Commercial")
                elif requested_type == "residential" and "residential" in proj_type_lower:
                    score += 10
                    matching_features.append("Property Type: Residential")
                else:
                    score -= 15

            # 2. BHK / Configuration Match (Weight: 15)
            if "bhk" in preferences:
                requested_bhk = preferences["bhk"]
                proj_config = details["Configurations"]
                matched = False
                if str(requested_bhk) in proj_config:
                    score += 15
                    reasons.append(f"Offers layout configuration matching {requested_bhk} BHK")
                    matching_features.append(f"Configuration: {requested_bhk} BHK Matched")
                    matched = True
                
                if not matched:
                    score -= 10

            # 3. Budget Match (Weight: 20)
            if "budget" in preferences:
                user_budget = preferences["budget"]
                min_p, max_p = self._parse_price_range(details["Price Range"])
                
                if min_p > 0:
                    if min_p <= user_budget <= max_p:
                        score += 20
                        reasons.append(f"Pricing fits within your target budget (INR {user_budget:.0f} Lakhs)")
                        matching_features.append(f"Price Match: Fits budget of {user_budget:.0f} Lakhs")
                    elif user_budget >= max_p:
                        score += 15
                        reasons.append("Excellent value choice (well below your budget ceiling)")
                        matching_features.append("Budget: Cost is below budget")
                    elif user_budget >= min_p * 0.9:  # within 10% stretch
                        score += 10
                        reasons.append("Slightly above budget (stretch of <10% required)")
                    else:
                        score -= 20

            # 4. Location Match (Weight: 20)
            if "locations" in preferences:
                matched_locs = []
                for loc_keyword in preferences["locations"]:
                    if loc_keyword in details["Location"].lower() or loc_keyword in details["City"].lower():
                        matched_locs.append(loc_keyword.title())
                if matched_locs:
                    score += 20
                    reasons.append(f"Located in target area: {', '.join(matched_locs)}")
                    matching_features.append(f"Location: {', '.join(matched_locs)} matched")
                else:
                    score -= 10

            # 5. Amenities Match (Weight: 10)
            if "amenities" in preferences:
                matched_amenities = []
                proj_amenities_lower = details["Amenities"].lower()
                for amenity in preferences["amenities"]:
                    if amenity in proj_amenities_lower:
                        matched_amenities.append(amenity.title())
                if matched_amenities:
                    score += len(matched_amenities) * 3
                    reasons.append(f"Includes amenities: {', '.join(matched_amenities[:3])}")
                    matching_features.append(f"Amenities: {', '.join(matched_amenities)}")
                else:
                    # Slight penalty if user explicitly asked for amenity not present
                    score -= 5

            # 6. Possession Match (Weight: 10)
            if "possession" in preferences:
                poss_year_match = re.search(r"\d{4}", details["Possession"])
                if poss_year_match:
                    year = int(poss_year_match.group(0))
                    if year <= 2026:
                        score += 10
                        reasons.append("Early delivery and near-term possession timelines")
                        matching_features.append(f"Possession: Early ({details['Possession']})")
                    else:
                        drawbacks.append("Distant possession and construction timelines")

            # 7. Builder Match (Weight: 10)
            if "builder" in preferences:
                requested_builder = preferences["builder"]
                if requested_builder in details["Builder"].lower():
                    score += 10
                    reasons.append(f"Developed by trusted builder: {details['Builder']}")
                    matching_features.append(f"Builder: {details['Builder']} matched")

            # 8. Investment intent: favour projects with higher appreciation potential
            if preferences.get("investment"):
                if "Hyderabad" in details.get("City", "") or "Bengaluru" in details.get("City", ""):
                    score += 10
                    reasons.append("Located in a high-appreciation metro corridor")
                    matching_features.append("Investment: High-growth city")
                if details["Property Type"] == "Commercial":
                    score += 5
                    reasons.append("Commercial asset class with strong lease yield potential")

            # 9. Proximity hints: no strict filtering, just boost schools/hospital-friendly cities
            if "proximity" in preferences:
                score += 5  # mild boost — all our projects are in established zones
                reasons.append("Located in an established urban zone with good civic infrastructure")

            # Clamp score between 0 and 100
            score = max(30.0, min(99.0, score))
            
            # Format reasons list
            if not reasons:
                reasons = ["Good match to your general requirements",
                           "Reputable developer & high-quality construction"]
            if not matching_features:
                matching_features = [f"Builder: {details['Builder']}",
                                     f"Location: {details['Location']}"]

            # Citations
            citations = []
            for doc in proj.documents:
                if "brochure" in doc.lower() or "rera" in doc.lower():
                    citations.append(f"{doc} | page 1")
            
            scored_list.append({
                "property_name": name,
                "project_name": name,
                "relevance_score": score / 100.0,
                "score_percentage": f"{score:.0f}%",
                "reasons": reasons[:3],
                "score": min(98.5, max(15.0, score)),
                "builder": details["Builder"],
                "price": details["Price Range"],
                "configuration": details["Configurations"],
                "reasons": reasons,
                "matching_features": matching_features,
                "citations": citations,
            })

        # Sort by relevance score descending
        return sorted(scored_list, key=lambda x: x["score"], reverse=True)

    def _parse_price_range(self, price_str: str) -> Tuple[float, float]:
        """Parse pricing bounds to float values in Lakhs.

        Handles mixed unit strings correctly, e.g.:
          "INR 78 lakh - 1.65 crore"  →  (78.0,  165.0)
          "INR 1.2 crore - 4.8 crore" →  (120.0, 480.0)
          "INR 52 lakh - 1.35 crore"  →  (52.0,  135.0)
        """
        price_clean = price_str.lower().strip()
        if not price_clean or price_clean == "unknown":
            return 0.0, 0.0

        # Split on dash/hyphen to handle each side independently
        dash_parts = re.split(r"\s*[-–]\s*", price_clean, maxsplit=1)

        def _to_lakhs(segment: str) -> float:
            """Convert a single price segment (e.g. '1.65 crore') to lakhs."""
            num_match = re.search(r"([\d.]+)", segment)
            if not num_match:
                return 0.0
            val = float(num_match.group(1))
            if "crore" in segment or " cr" in segment:
                val *= 100.0
            # lakh / l → already in lakhs
            return val

        if len(dash_parts) == 2:
            min_val = _to_lakhs(dash_parts[0])
            max_val = _to_lakhs(dash_parts[1])
            # If only one side has a unit, inherit from the other
            if min_val == 0.0:
                min_val = max_val
            if max_val == 0.0:
                max_val = min_val
            return min_val, max_val
        else:
            val = _to_lakhs(price_clean)
            return val, val
