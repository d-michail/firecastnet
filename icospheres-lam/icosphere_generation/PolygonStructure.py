from typing import Dict, Any, Optional
from shapely.geometry import Polygon

DEFAULT_BUFFER_FACTOR = 50.0
DEFAULT_REFINEMENT_TYPE = "none"
DEFAULT_BUFFER_UNIT = "km"

class PolygonStructure:
    """Type definition for polygon structure used in icosphere generation."""
    
    def __init__(self, 
                 target_code: str,
                 refinement_order: int,
                 refinement_type: str = DEFAULT_REFINEMENT_TYPE,
                 buffer_factor: float = DEFAULT_BUFFER_FACTOR,
                 buffer_unit: str = DEFAULT_BUFFER_UNIT,
                 interest: bool = True,
                 wkt: Optional[Polygon] = None):
        self.target_code = target_code
        self.refinement_order = refinement_order
        self.refinement_type = refinement_type
        self.buffer_factor = buffer_factor
        self.buffer_unit = buffer_unit
        self.interest = interest
        self.wkt = wkt
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "target_code": self.target_code,
            "refinement_order": self.refinement_order,
            "refinement_type": self.refinement_type,
            "buffer_factor": self.buffer_factor,
            "buffer_unit": self.buffer_unit,
            "interest": self.interest
        }
        if self.wkt is not None:
            result["wkt"] = self.wkt
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PolygonStructure':
        """Create from dictionary representation."""
        return cls(
            target_code=data["target_code"],
            refinement_order=data["refinement_order"],
            refinement_type=data.get("refinement_type", DEFAULT_REFINEMENT_TYPE),
            buffer_factor=data.get("buffer_factor", DEFAULT_BUFFER_FACTOR),
            buffer_unit=data.get("buffer_unit", DEFAULT_BUFFER_UNIT),
            interest=data.get("interest", True),
            wkt=data.get("wkt")
        )
    
    def copy(self) -> 'PolygonStructure':
        """Create a copy of this PolygonStructure."""
        return PolygonStructure(
            target_code=self.target_code,
            refinement_order=self.refinement_order,
            refinement_type=self.refinement_type,
            buffer_factor=self.buffer_factor,
            buffer_unit=self.buffer_unit,
            interest=self.interest,
            wkt=self.wkt
        )
    
    def __str__(self) -> str:
        """String representation of PolygonStructure."""
        wkt_info = f", wkt={type(self.wkt).__name__}" if self.wkt else ", wkt=None"
        return (f"PolygonStructure(target_code='{self.target_code}', "
                f"refinement_order={self.refinement_order}, "
                f"refinement_type='{self.refinement_type}', "
                f"buffer_factor={self.buffer_factor}, "
                f"buffer_unit='{self.buffer_unit}', "
                f"interest={self.interest}{wkt_info})")