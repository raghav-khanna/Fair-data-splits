from classes.PipelineRunner import PipelineRunnerClass
import warnings
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category = UndefinedMetricWarning)


def all_functions(name: str) -> int:
    # print('Import all functions to this file and run only this file: ', end = '')
    # print(name)

    # Dataset settings
    file_path = '<repository absolute path>/Fair-data-splits/data/student-mat.csv'
    columns_to_hot_encode = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian']
    yn_tf_columns = ['schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']
    target_column_name = 'G3'
    columns_to_remove_right_before_classification = ['G1', 'G2']
    sensitive_attribute_column = 'sex_is_F'
    sensitive_attribute_value = 1

    # Parameter settings
    minimum_correlation_threshold = 0.1
    classifier_name = 'SVC'
    split_ratio = 0.25
    iterations = 100
    fair_metric_list = [
        'statistical_parity',
        # 'conditional_parity',
        'equalized_odds', 'equal_opportunity', 'predictive_equality', 'conditional_use_accuracy_equality', 'predictive_parity', 'overall_accuracy_equality'
    ]
    pf = PipelineRunnerClass(
        file_path = file_path,
        columns_to_hot_encode = columns_to_hot_encode,
        yn_tf_columns = yn_tf_columns,
        target_column_name = target_column_name,
        minimum_correlation_threshold = minimum_correlation_threshold,
        columns_to_remove_right_before_classification = columns_to_remove_right_before_classification,
        classifier_name = classifier_name,
        sensitive_attribute_value = sensitive_attribute_value,
        sensitive_attribute_column = sensitive_attribute_column,
        split_ratio = split_ratio,
        iterations = iterations,
        fair_metrics_list = fair_metric_list
    )
    pf.run_classification_pipeline()
    return 1


if __name__ == "__main__":
    all_functions('run.py')
