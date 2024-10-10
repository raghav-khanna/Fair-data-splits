from classes.PipelineRunner import PipelineRunnerClass


def all_functions(name: str) -> int:
    # print('Import all functions to this file and run only this file: ', end = '')
    # print(name)
    data_path = '/Users/pranavchatur/Downloads/student+performance/student/student-mat.csv'
    columns_to_hot_encode = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian']
    yn_tf_columns = ['schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']
    target_column_name = 'G3'
    minimum_correlation_threshold = 0.1
    columns_to_remove_right_before_classification = None
    use_model = 'SVC'
    pf = PipelineRunnerClass(data_path = data_path, columns_to_hot_encode = columns_to_hot_encode, yn_tf_columns = yn_tf_columns, target_column_name = target_column_name, minimum_correlation_threshold = minimum_correlation_threshold, columns_to_remove_right_before_classification = columns_to_remove_right_before_classification, use_model = use_model)
    pf.run_classification_pipeline()
    return 1


if __name__ == "__main__":
    all_functions('run.py')
