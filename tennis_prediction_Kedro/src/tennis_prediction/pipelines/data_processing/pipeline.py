from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    process_data,
    feature_engineering,
    encoder,
    scaler,

)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=process_data, 
                inputs='atp_tennis', 
                outputs='df_top100', 
                name='process_data_node'
            ),

            node(func=feature_engineering, 
                 inputs='df_top100', 
                 outputs='df_top100_features', 
                 name='feature_engineering_node'
            ),

            node(func=encoder, 
                 inputs='df_top100_features', 
                 outputs='df_top100_encoder', 
                 name='encoder_variables_node'
            ),

            node(func=scaler, 
                 inputs='df_top100_encoder', 
                 outputs='df_top100_scaler', 
                 name='scaler_variables_node'
            ),

    ])



     