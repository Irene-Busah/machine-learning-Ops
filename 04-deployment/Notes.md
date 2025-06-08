## Machine Learning Deployment Notes
Machine learning deployment involves taking a trained model and making it available for use in production environments. This can include serving the model via APIs, integrating it into applications, or deploying it on edge devices.

Model deployment can be batch offline deployment or streaming or online.
Batch offline deployment involves running the model on a schedule to process large datasets. The model runs regularly, such as daily or weekly, to generate predictions for a batch of data.

Streaming or online deployment involves serving the model in real-time, allowing it to make predictions on incoming data as it arrives. This is often done through APIs that applications can call to get predictions. This type of deployment can be done through **web services or streaming services**. The choice between using web services or streaming services depends on the use case and the nature of the data being processed.


### Deploying Models with Web Services
