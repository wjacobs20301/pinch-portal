# Model files (`models/*.onnx`)

The anime models were trained on image pairs derived from the
[HaGRID / HaGRIDv2 dataset](https://github.com/ai-forever/hagrid) (ai-forever / SberDevices), which is published
under a variant of the Creative Commons Attribution-ShareAlike 4.0 licence
([licence text](https://github.com/ai-forever/hagrid/blob/master/license/en_us.pdf)). To respect its attribution and
share-alike terms, the model files are released under **CC BY-SA 4.0**
(<https://creativecommons.org/licenses/by-sa/4.0/>), with attribution to HaGRID and this project.

The training targets were generated with Stable Diffusion 1.5-based models (Counterfeit V3, CreativeML OpenRAIL-M)
and ControlNets (lineart; CrucibleAI ControlNetMediaPipeFace, OpenRAIL). Please also follow the use restrictions of
the CreativeML OpenRAIL-M licence (<https://huggingface.co/spaces/CompVis/stable-diffusion-license>) — e.g. no use
to harm, impersonate or deceive people.

Changes: the HaGRID photos were subsampled, cropped/resized, redrawn by the teacher models and used only as
training data; no photos are included in this repository.
