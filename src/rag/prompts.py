GRAPH_EXTRACTION_PROMPT = """
-Goal-
Given a text document that describes a Process Engineering System (P&ID), identify all entities (Equipment, Piping, Instruments) and their relationships (Connections, Flows, Signals).

-Steps-
1. Identify all entities. For each entity, extract the following information:
- entity_name: Name of the entity, capitalized (e.g., "Pump P-101")
- entity_type: One of [Equipment, Piping, Instrument, Signal, Process]
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity" <|> <entity_name> <|> <entity_type> <|> <entity_description>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair, extract the following information:
- source_entity: name of the source entity
- target_entity: name of the target entity
- relationship_description: explanation of why you think they are related (e.g., connected_to, flows_into, measures)
- relationship_strength: a numeric score (1-10) indicating strength of the relationship
Format each relationship as ("relationship" <|> <source_entity> <|> <target_entity> <|> <relationship_description> <|> <relationship_strength>)

-Real Data Examples-
Example 1:
Text: In DEXPI file C01.xml, Pump P-101 is a Centrifugal Pump connected to Tank T-500 via Pipe Seg-400.
Output:
("entity" <|> "Pump P-101" <|> "Equipment" <|> "Centrifugal Pump defined in C01.xml")
("entity" <|> "Tank T-500" <|> "Equipment" <|> "Storage Tank defined in C01.xml")
("entity" <|> "Pipe Seg-400" <|> "Piping" <|> "Piping segment connecting P-101 to T-500")
("relationship" <|> "Pump P-101" <|> "Pipe Seg-400" <|> "connected_to" <|> 10)
("relationship" <|> "Pipe Seg-400" <|> "Tank T-500" <|> "flows_into" <|> 10)

Example 2:
Text: Valve V-101 is controlled by Signal S-202 which originates from Controller C-900.
Output:
("entity" <|> "Valve V-101" <|> "Equipment" <|> "Control Valve")
("entity" <|> "Signal S-202" <|> "Signal" <|> "Control signal for V-101")
("entity" <|> "Controller C-900" <|> "Instrument" <|> "Controller unit")
("relationship" <|> "Controller C-900" <|> "Signal S-202" <|> "sends_signal" <|> 9)
("relationship" <|> "Signal S-202" <|> "Valve V-101" <|> "controls" <|> 10)

-output-
"""