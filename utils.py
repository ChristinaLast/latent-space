
def list_all_subdirectories(base_path):
    subdirectories = []
    for root, dirs, files in os.walk(base_path):
        if not dirs:  # This means it's a leaf directory
            subdirectories.append(root)
    return subdirectories

masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)