# xflow_project
mlops project using airflow and mlflow

1. Создал виртульное окружение, файловую структуру, скрипты. Установил и настроил airflow и mlflow:

   ![image](https://github.com/med-alex/xflow_project/assets/118723191/cef8f865-8451-47f0-8fa1-9596967ef92a)
   
2. Реализовал DAG:
   
   ![image](https://github.com/med-alex/xflow_project/assets/118723191/e7fc11a9-97db-463f-bd10-1dbc7e0f2d84)
   
3. Добавл в скрипты mlflow. Сохраняются артифакты c моделью, параметрами и скором. В тестировании модели модель подгружается из mlflow, а не из папки с помощью pikle.

   ![image](https://github.com/med-alex/xflow_project/assets/118723191/b0645e52-274a-4332-a08d-42927863f03f)
   ![image](https://github.com/med-alex/xflow_project/assets/118723191/552f034d-ff23-4c3b-89d0-b575ca1e4ec2)

4. Запуски airflow:

   ![image](https://github.com/med-alex/xflow_project/assets/118723191/a938fb95-9e3a-4b81-b735-47f06b21dd21)

5. Результаты в mlflow:

   ![image](https://github.com/med-alex/xflow_project/assets/118723191/edf6d340-3a60-497f-8f82-3066e2027a5f)
