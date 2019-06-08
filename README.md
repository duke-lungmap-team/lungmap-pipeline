# lungmap-pipeline
Complete analysis pipeline for the segmentation &amp; classification of structures / cells for LungMap images


## How to Use
This repository is meant to be a demonstration of how the [microscopy-analysis-pipeline](https://github.com/duke-lungmap-team/microscopy-analysis-pipeline)
can be used to integrate with [lungmap data](https://github.com/duke-lungmap-team/lungmap-image-data) to effect
instance segmentation and classification of anatomical structures. As such, we provide several [python scripts](examples)
which demonstrate how to use the resources together. But first, to run the scripts, a little setup...

### Environment
We assume a Python 3 environment with dependencies listed in the `requirements.txt` file.
```
pip install -r requirements.txt
```

### Data
For the [python scripts](examples) to run properly, we assume that the [lungmap data](https://github.com/duke-lungmap-team/lungmap-image-data)
has been downloaded and symlinked as `data` into this repositories parent folder.
```
cd ~
git clone https://github.com/duke-lungmap-team/lungmap-image-data.git
cd /path/to/lungmap-pipeline
ln -s ~/lungmap-image-data data
```

### Running `example/*.py`
Check the top of each script before running. Some scripts assume that other scripts have been run first. This is 
done to stash model objects or other metadata needed to produce further output.