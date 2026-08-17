import json
from pathlib import Path


def extract_uids_from_objectbox_json(json_file_path):
    """
    Extract UIDs from ObjectBox default.json file and generate Python entity code
    """
    with open(json_file_path) as f:
        data = json.load(f)

    python_code = []

    for entity in data.get("entities", []):
        # Extract entity info
        entity_id_uid = entity["id"]  # Format: "ID:UID"
        entity_id, entity_uid = entity_id_uid.split(":")
        entity_name = entity["name"]

        # Extract dimensions from entity name (e.g., VectorEntity384 -> 384)
        dimensions = "".join(filter(str.isdigit, entity_name))

        # Start entity definition
        python_code.append(f"@Entity(uid={entity_uid})")
        python_code.append(f"class {entity_name}:")

        # Extract property UIDs
        for prop in entity.get("properties", []):
            prop_id_uid = prop["id"]  # Format: "ID:UID"
            prop_id, prop_uid = prop_id_uid.split(":")
            prop_name = prop["name"]

            # Generate property definition based on name
            if prop_name == "id":
                python_code.append(f"    {prop_name} = Id(id={prop_id}, uid={prop_uid})")
            elif prop_name == "embedding":
                # Check if property has indexId (HnswIndex)
                if "indexId" in prop:
                    python_code.append(
                        f"    {prop_name} = Float32Vector(id={prop_id}, uid={prop_uid}, index=HnswIndex(dimensions={dimensions}, distance_type=VectorDistanceType.COSINE))"
                    )
                else:
                    python_code.append(f"    {prop_name} = Float32Vector(id={prop_id}, uid={prop_uid})")
            elif prop_name == "timestamp":
                python_code.append(f"    {prop_name} = Property(int, id={prop_id}, uid={prop_uid})")
            elif prop_name == "content":
                # Always add index for content field
                if "indexId" in prop:
                    index_id, index_uid = prop["indexId"].split(":")
                    python_code.append(
                        f"    {prop_name} = String(id={prop_id}, uid={prop_uid}, index=Index(type=IndexType.VALUE, uid={index_uid}))"
                    )
                else:
                    python_code.append(
                        f"    {prop_name} = String(id={prop_id}, uid={prop_uid}, index=Index(type=IndexType.VALUE))"
                    )
            else:
                # Other string properties (name, document, metadata)
                if "indexId" in prop:
                    index_id, index_uid = prop["indexId"].split(":")
                    python_code.append(
                        f"    {prop_name} = String(id={prop_id}, uid={prop_uid}, index=Index(type=IndexType.VALUE, uid={index_uid}))"
                    )
                else:
                    python_code.append(f"    {prop_name} = String(id={prop_id}, uid={prop_uid})")

        python_code.append("")  # Empty line between entities

    return "\n".join(python_code)


def extract_uids_for_kotlin(json_file_path):
    """
    Extract UIDs and generate Kotlin @Uid annotations
    """
    with open(json_file_path) as f:
        data = json.load(f)

    kotlin_annotations = []

    for entity in data.get("entities", []):
        entity_id_uid = entity["id"]
        entity_id, entity_uid = entity_id_uid.split(":")
        entity_name = entity["name"]

        # Extract dimensions from entity name
        dimensions = "".join(filter(str.isdigit, entity_name))

        kotlin_annotations.append(f"// {entity_name}")
        kotlin_annotations.append("@Entity")
        kotlin_annotations.append(f"@Uid({entity_uid})")
        kotlin_annotations.append(f"data class {entity_name}(")

        for prop in entity.get("properties", []):
            prop_id_uid = prop["id"]
            prop_id, prop_uid = prop_id_uid.split(":")
            prop_name = prop["name"]

            if prop_name == "id":
                kotlin_annotations.append(f"    @Id @Uid({prop_uid}) override var {prop_name}: Long = 0,")
            elif prop_name == "embedding":
                kotlin_annotations.append(
                    f"    @HnswIndex(dimensions = {dimensions}, distanceType = VectorDistanceType.COSINE)"
                )
                kotlin_annotations.append(
                    f"    @Uid({prop_uid}) override var {prop_name}: FloatArray = floatArrayOf(),"
                )
            elif prop_name == "content":
                # Always add @Index for content field
                kotlin_annotations.append(
                    f'    @Index @Uid({prop_uid}) override var {prop_name}: String = "",'
                )
            else:
                if prop_name == "timestamp":
                    kotlin_annotations.append(
                        f"    @Uid({prop_uid}) override var {prop_name}: Long = System.currentTimeMillis(),"
                    )
                else:
                    kotlin_annotations.append(f'    @Uid({prop_uid}) override var {prop_name}: String = "",')

        kotlin_annotations.append(")")
        kotlin_annotations.append("")

    return "\n".join(kotlin_annotations)


def create_filtered_json_model(
    json_file_path, entity_names_to_include, output_path="objectbox-model/default.json"
):
    """
    Create a filtered ObjectBox JSON model that only includes specified entities
    while preserving their original IDs and UIDs
    """
    import os

    with open(json_file_path) as f:
        data = json.load(f)

    # Filter entities to only include specified ones
    filtered_entities = []
    for entity in data.get("entities", []):
        if entity["name"] in entity_names_to_include:
            filtered_entities.append(entity)

    # Create filtered model
    filtered_data = data.copy()
    filtered_data["entities"] = filtered_entities

    # Update lastEntityId to the highest ID among included entities
    if filtered_entities:
        last_entity = max(filtered_entities, key=lambda e: int(e["id"].split(":")[0]))
        filtered_data["lastEntityId"] = last_entity["id"]

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Write filtered model
    with open(output_path, "w") as f:
        json.dump(filtered_data, f, indent=2)

    print(f"✅ Filtered ObjectBox model saved to '{output_path}'")
    print(f"📋 Included entities: {entity_names_to_include}")

    return output_path


# Example usage:
if __name__ == "__main__":
    # Replace with your actual path
    # S9: was the repo-relative "database/default.json". That root is gone and this module now
    # ships inside the wheel, so resolve the schema beside the module instead of beside the CWD.
    json_path = str(Path(__file__).parent / "default.json")

    print("=== PYTHON ENTITIES ===")
    python_entities = extract_uids_from_objectbox_json(json_path)
    print(python_entities)

    print("\n=== KOTLIN ANNOTATIONS ===")
    print(extract_uids_for_kotlin(json_path))

    # Save Python entities to file
    with open("vector_entity.py", "w") as f:
        f.write("from objectbox.model import *\n")
        f.write("from objectbox.model.properties import Index, IndexType\n\n")
        f.write("# Auto-generated ObjectBox entities with UIDs from Kotlin\n")
        f.write("# Generated from: " + json_path + "\n\n")
        f.write(python_entities)

    print("\n✅ Python entities saved to 'vector_entity.py'")

    # Create filtered model for Python (example: only include VectorEntity384)
    entities_to_use = ["VectorEntity384"]  # Modify this list as needed
    create_filtered_json_model(json_path, entities_to_use)
