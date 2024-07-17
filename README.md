# Modeling Scattered Light in TESS Full Frame Images Using Generative AI

This project use machine learning models to predict the shape and intensity of scattered light in Transiting Exoplanet Survey Satellite (TESS) full frame images (FFIs).

## Folders

* data
  * contains dataset information, folders, and preprocessing pipelines that all models use
* model_conditional_diffusion
  * Conditional diffusion models for TESS dataset and MNIST test model
* model_conditional_norm_flow
  * Conditional normalizing flows model for TESS dataset
* model_conditional_norm_flow_horses
  * Conditional normalizing flows model for testing the model on the Weizmann horse database
* model_light
  * Dense neural network model for TESS dataset
* model_norm_flow
  * Normalizing flows model (no conditional) for TESS dataset
 
## Acknowledgments

I would like to acknowledge the TESS team for their assistance througout this project, expecially Dr. Daniel Muthukrishna and Dr. Roland Vanderspek.
