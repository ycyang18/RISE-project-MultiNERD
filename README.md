## Resources

- MultiNERD Named Entity Recognition dataset: https://huggingface.co/datasets/Babelscape/multinerd?row=17
- Accompanying paper: https://aclanthologv.org/2022.findings-naacl.60.pdf
- Please feel free to use any suitable compute resource to which you have access. you do not have access to suitable resources, you may wish to consider using Google Colab: https://colab.google/

## Instructions

Using the MultiNERD Named Entity Recognition (NER) dataset, complete the following steps to train and evaluate a Named Entity Recognition model for English: 
1. Familiarize yourself with the dataset and the task. Find a suitable LM model on HuggingFace Model Hub (https://huggingface.co/models). This can be a Large Language Model (LLM) or any type of Transformer-based Language Model
2. Filter out the non-EngIish examples of the dataset
3. Fine-tune your chosen model on the English subset of the training set. This will be system A
4. You will now train a model that will predict only five entity types and the O tag (i.e. not part of an entity). Therefore, you should perform the necessay pre-processing steps on the dataset. All examples should thus remain, but entity types not belonging to one of the following five should be set to zero: PERSON(PER), ORGANIZATION(ORG), LOCATION(LOC), DISEASES(DIS), ANIMAL(ANIM)
5. Fine-tune your model on the filtered dataset that you constructed in Step 5. This will be system B
6. Pick a suitable metric or suite of metrics and evaluate both systems A and B using the test set
7. Create a new Github repository and upload your code solution. You should include a README file with instructions on how to run your code, and a requirements.txt file to create the environment needed to run your code
8. Write a short paragraph (maximum 200 words) highlighting the main findings of your experiments as well as any limitations of your approach

## Todo
1. Adjust the mapping of filtered entities