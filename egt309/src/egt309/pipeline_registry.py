"""Project pipelines."""
from kedro.pipeline import Pipeline
from egt309.pipelines import data_prep as dp
from egt309.pipelines import data_science as ds
#from egt309.pipelines import reporting as rp

def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to `Pipeline` objects.
    """
    
    # Define the Pipeline Objects
    data_prep_pipeline = dp.create_pipeline()
    data_science_pipeline = ds.create_pipeline()
    #reporting_pipeline = rp.create_pipeline()

    return {
<<<<<<< HEAD
        # Register the Default Pipeline (runs everything)
        "__default__": data_prep_pipeline + data_science_pipeline, 
=======
        # egister the Default Pipeline (runs everything)
        "__default__": data_prep_pipeline + data_science_pipeline,
>>>>>>> 8583e0b312d2cfea10367e621b2598e26c4c9a0b
        
        # Register the Individual Pipelines for targeted runs
        "data_prep": data_prep_pipeline,
        "data_science": data_science_pipeline,
        #"reporting": reporting_pipeline
    }