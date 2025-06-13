from kedro.pipeline import Pipeline, node, pipeline
from .nodes import split_data, train_model, evaluate_model

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                inputs=dict(data="df_top100_scaler", parameters="params:modeling"),
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data_node"
            ),
            node(
                func=train_model,
                inputs=["X_train", "y_train", "X_test"],
                outputs=["model", "y_pred"],
                name="train_model_node"
            ),
            node(
                func=evaluate_model,
                inputs=["model", "y_test", "y_pred"],
                outputs="metrics",
                name="evaluate_model_node"
            )
        ]
    )
