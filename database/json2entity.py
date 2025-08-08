import json

def extract_uids_from_objectbox_json(json_file_path):
    """
    Extract UIDs from ObjectBox default.json file and generate Python entity code
    """
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    python_code = []
    
    for entity in data.get('entities', []):
        # Extract entity info
        entity_id_uid = entity['id']  # Format: "ID:UID"
        entity_id, entity_uid = entity_id_uid.split(':')
        entity_name = entity['name']
        
        # Extract dimensions from entity name (e.g., VectorEntity384 -> 384)
        dimensions = ''.join(filter(str.isdigit, entity_name))
        
        # Start entity definition
        python_code.append(f"@Entity(uid={entity_uid})")
        python_code.append(f"class {entity_name}:")
        
        # Extract property UIDs
        for prop in entity.get('properties', []):
            prop_id_uid = prop['id']  # Format: "ID:UID"
            prop_id, prop_uid = prop_id_uid.split(':')
            prop_name = prop['name']
            
            # Generate property definition based on name
            if prop_name == 'id':
                python_code.append(f"    {prop_name} = Id(id={prop_id}, uid={prop_uid})")
            elif prop_name == 'embedding':
                # Check if property has indexId (HnswIndex)
                if 'indexId' in prop:
                    python_code.append(f"    {prop_name} = Float32Vector(id={prop_id}, uid={prop_uid}, index=HnswIndex(dimensions={dimensions}, distance_type=VectorDistanceType.COSINE))")
                else:
                    python_code.append(f"    {prop_name} = Float32Vector(id={prop_id}, uid={prop_uid})")
            elif prop_name == 'timestamp':
                python_code.append(f"    {prop_name} = Property(int, id={prop_id}, uid={prop_uid})")
            else:
                python_code.append(f"    {prop_name} = String(id={prop_id}, uid={prop_uid})")
        
        python_code.append("")  # Empty line between entities
    
    return "\n".join(python_code)

def extract_uids_for_kotlin(json_file_path):
    """
    Extract UIDs and generate Kotlin @Uid annotations
    """
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    kotlin_annotations = []
    
    for entity in data.get('entities', []):
        entity_id_uid = entity['id']
        entity_id, entity_uid = entity_id_uid.split(':')
        entity_name = entity['name']
        
        kotlin_annotations.append(f"// {entity_name}")
        kotlin_annotations.append(f"@Entity")
        kotlin_annotations.append(f"@Uid({entity_uid})")
        kotlin_annotations.append(f"data class {entity_name}(")
        
        for prop in entity.get('properties', []):
            prop_id_uid = prop['id']
            prop_id, prop_uid = prop_id_uid.split(':')
            prop_name = prop['name']
            
            if prop_name == 'id':
                kotlin_annotations.append(f"    @Id @Uid({prop_uid}) override var {prop_name}: Long = 0,")
            elif prop_name == 'embedding':
                kotlin_annotations.append(f"    @HnswIndex(dimensions = XXX, distanceType = VectorDistanceType.COSINE)")
                kotlin_annotations.append(f"    @Uid({prop_uid}) override var {prop_name}: FloatArray = floatArrayOf(),")
            else:
                if prop_name == 'timestamp':
                    kotlin_annotations.append(f"    @Uid({prop_uid}) override var {prop_name}: Long = System.currentTimeMillis(),")
                else:
                    kotlin_annotations.append(f"    @Uid({prop_uid}) override var {prop_name}: String = \"\",")
        
        kotlin_annotations.append(")")
        kotlin_annotations.append("")
    
    return "\n".join(kotlin_annotations)

# Example usage:
if __name__ == "__main__":
    # Replace with your actual path
    json_path = "database/default.json"
    
    print("=== PYTHON ENTITIES ===")
    python_entities = extract_uids_from_objectbox_json(json_path)
    
    print("\n=== KOTLIN ANNOTATIONS ===")
    print(extract_uids_for_kotlin(json_path))

    # Save Python entities to file
    with open("vector_entity.py", "w") as f:
        f.write("from objectbox.model import *\n\n")
        f.write("# Auto-generated ObjectBox entities with UIDs from Kotlin\n")
        f.write("# Generated from: " + json_path + "\n\n")
        f.write(python_entities)
    
    print(f"\n✅ Python entities saved to 'vector_entity.py'")