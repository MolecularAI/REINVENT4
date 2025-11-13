names, scoring_function, transforms, weights = params

component_results = compute_component_scores(
    smilies, scoring_function, {}, valid_mask  # Pass empty dict to disable cache
)

transformed_scores = []
