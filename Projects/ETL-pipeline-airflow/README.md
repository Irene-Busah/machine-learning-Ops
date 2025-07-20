## Project Overview - Airflow ETL Pipeline with Postgres and API Integration

This project involves creating an ETL (Extract, Transform, Load) pipeline using Apache Airflow. The pipeline extracts data from an external API, in this case, NASA's Astronomy Picture of the Day, transform the data, and loads it into a Postgres database. The entire workflow is orchestrated by Airflow, a platform thatallows scheduling, monitoring, and managing workflows.


The project leverages Docker to run Airflow and Postgres as services, ensuring an isolated and reproducible environment. We also utilize Airflow hooks and operators to handle the ETL process efficiently.
