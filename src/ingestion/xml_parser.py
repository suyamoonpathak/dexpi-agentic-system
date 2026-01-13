import logging
from pathlib import Path
from typing import List, Dict, Any
from lxml import etree

logger = logging.getLogger(__name__)

class DexpiParser:
    """
    Parses DEXPI P&ID XML files into a structured dictionary.
    Focuses on extracting Equipment, Piping, and their attributes.
    """

    def _strip_namespaces(self, tree: etree._ElementTree) -> etree._ElementTree:
        """Removes XML namespaces to simplify tag access."""
        for elem in tree.iter():
            if not isinstance(elem.tag, str):
                continue
            if '}' in elem.tag:
                elem.tag = elem.tag.split('}', 1)[1]
        return tree

    def parse_file(self, file_path: str) -> Dict[str, Any]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"Parsing DEXPI file: {path.name}")
        
        try:
            parser = etree.XMLParser(recover=True)
            tree = etree.parse(str(path), parser)
            tree = self._strip_namespaces(tree)
            root = tree.getroot()
            
            equipment = self._extract_equipment(root)
            connections = self._extract_connections(root)
            
            logger.info(f"Extracted {len(equipment)} items and {len(connections)} connections.")
            
            return {
                "filename": path.name,
                "equipment": equipment,
                "connections": connections
            }
        except Exception as e:
            logger.error(f"Failed to parse {path.name}: {str(e)}")
            raise e

    def _extract_equipment(self, root: etree._Element) -> List[Dict[str, Any]]:
        items = []
        # Filter out graphical/metadata noise
        IGNORED_TYPES = {
            "DexpiLabel", "Label", "Text", "Presentation", "Shape", 
            "PropertyBreak", "DrawingBorder", "PlotBorder", "MetaData", 
            "PlantStructureItem", "DrawingWrapper"
        }

        for elem in root.xpath("//*[@ComponentClass]"):
            tag_type = elem.get("ComponentClass")
            if tag_type in IGNORED_TYPES:
                continue

            attrs = self._get_generic_attributes(elem)
            
            # Heuristic for finding the human-readable tag
            raw_tag = elem.get("ComponentName") or elem.get("TagName")
            if not raw_tag or raw_tag == "Unknown":
                raw_tag = attrs.get("SegmentNumberAssignmentClass") or \
                          attrs.get("PipingComponentNumberAssignmentClass") or \
                          attrs.get("InstrumentationLoopFunctionNumberAssignmentClass") or \
                          "Unnamed_Component"

            items.append({
                "id": elem.get("ID"),
                "tag": raw_tag,
                "type": tag_type,
                "attributes": attrs
            })
        return items

    def _extract_connections(self, root: etree._Element) -> List[Dict[str, Any]]:
        connections = []
        
        # 1. Explicit Connections
        for conn in root.xpath("//Connection"):
            src = conn.get("FromID") or conn.get("SourceID")
            tgt = conn.get("ToID") or conn.get("TargetID")
            if src and tgt:
                connections.append({"source": src, "target": tgt, "type": "ConnectedTo"})

        # 2. Piping Segments (Implicit Connections)
        for segment in root.xpath("//PipingNetworkSegment"):
            seg_id = segment.get("ID")
            for port in segment.xpath(".//ConnectionPoint"):
                node_id = port.get("NodeID")
                if node_id:
                     connections.append({"source": node_id, "target": seg_id, "type": "PipeConnection"})

        return connections

    def _get_generic_attributes(self, elem: etree._Element) -> Dict[str, str]:
        """Flattens <GenericAttribute> tags into a dictionary."""
        attrs = {}
        for ga in elem.xpath(".//GenericAttribute"):
            name = ga.get("Name")
            value = ga.get("Value")
            if name and value:
                attrs[name] = value
        return attrs