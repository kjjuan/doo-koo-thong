"""Project pipelines."""
from kedro.pipeline import Pipeline
from egt309.pipelines import data_prep as dp
from egt309.pipelines import data_science as ds

def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to `Pipeline` objects.
    """
    
    # Define the Pipeline Objects
    data_prep_pipeline = dp.create_pipeline()
    data_science_pipeline = ds.create_pipeline()

    return {
        # Register the Default Pipeline (runs everything)
        "__default__": data_prep_pipeline + data_science_pipeline, 
        
        # Register the Individual Pipelines for targeted runs
        "data_prep": data_prep_pipeline,
        "data_science": data_science_pipeline
    }