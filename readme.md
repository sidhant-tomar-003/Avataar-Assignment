# Auto-Segmentation CLI
A CLI tool that has the flexibility of a text prompt to select masks from a given image
- Uses YOLO to accelerate detection of common objects
- Less common objects are detected using SAM2 auto segmentation + CLIP
- Allows for multiple masks of target class
- Easy set up, easy usage 

## The basic idea:
- SAM2 has incredible performance when provided a point to segment off of.
- Using that in conjunction with its auto-annotation, almost any notable feature in an image can be segmented satisfactorily.
- The challange comes in classifying the segments, but that can be done with the help of CLIP.
- However, this approach cannot be done in a zero-shot very large vocabulary way.
- Without any fine tuning or lowering the potential answers, CLIP is unable to perform too well.
- This can be remedied by picking specific fine-tuned models based on context.
- There are many other heuristic-based approaches to improving this tool. Please do suggest any if you have some! 


## How to run:
- run the python script "init_project.py" to install the required models and libraries
- Then, run the CLI tool by using the following syntax:
    - python run1.py --image ./example.jpg --class "chair" --output ./generated.png
- Reccomended to run in kaggle VMs for simplicity
- I would also reccomend checking out the experimenting.ipynb to more easily follow the pipeline.



## Requirements:
- Please use a CUDA enabled GPU for maximum performance



## Todo:
There are several planned improvements.
- Using fine tuned CLIP models based on input class semantical classification
- More sophisticated prompting to enable better CLIP inferencing
- Larger dictionary size (currently built to match photos in images)
- More sophisticated actions like pose editing


### Contributing:
- Anyone is free to contribute however they wish
- Also, feel free to drop suggestions! 
- Feel free to reach out to me on discord @sid_the_slotth, cheers
- Note that this repository is slated to be changed a lot with revamps that will not have backwards compatibility in mind. 