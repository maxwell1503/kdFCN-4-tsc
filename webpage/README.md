# kdFCN-4-tsc companion page template

This is the template of the companion page available [here](https://maxime-devanne.com/pages/kdFCN-4-tsc/).
This template is inspired from [this repo](https://github.com/nerfies/nerfies.github.io).

**If you use these templates, please to refer to both links in the footer of your page.**

## Dynamic content

We added some dynamic content features through json and bibtex files.
To adapt the template to a paper, you just have to modify:

1) The [paper_data.json](static/jsons/paper_data.json) file:

* title: title of the paper
* authors: list of the authors including for each the name, the link and the institution_id from the list of institutions
* institutions: list of the institutions
* pdf_link (optional): link to the paper in pdf
* doi_link (optional): link to the doi
* slides_link (optional): link to the oral presentation
* code_link (optional): link to the code
* abstract: abstract text of the paper
* image_overview: path to the overview figure
* caption_overview: caption text of the overview figure
* external_htmls (optional): path to external htmls that will be included after the abstract to present your specific results related to your paper
* acknowledgment (optional): acknowledgment text

2) The [bibtex.bib](static/bibtex/bibtex.bib) file: 


