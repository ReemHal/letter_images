# Letter Images
Letters are sometimes saved as images. This repo extracts information from the image of a business letter such as: sender, recipient, date, signator, enclosures, and letter body. It also extracts and saves layout and content information of all images in a given dataset. The saved information can then be retrieved into a Pandas dataframe for further search on the contents of those letters.

Examples of how we can use the repo are given in the two iPython notebooks: __[explore_letter_image_dataset.ipynb](https://github.com/ReemHal/letter_images/blob/master/explore_letter_image_dataset.ipynb)__ and __[letter_image_examples.ipynb](https://github.com/ReemHal/letter_images/blob/master/letter_image_examples.ipynb)__.

## How to use the repo

### Extract information from a single letter image

To extract layout and content information from a single letter image, import letter_images into your script. This class handles letters saved as images.  Functions in this class identify blocks of text in the image, extract text within each block, identify different parts of the letter as sender/header, recipient, date, subject, letter boday, signature, and notes/enclosures. There are also functions to display the contours in the image, display the image, print  out the raw text within the image (without segmenting it into blocks), and display the text within each part of the processed letter.

#### Example

For example, given the letter image 

![sample letter](https://media.gcflearnfree.org/content/596f931e8444e81d1ca6cdfd_07_19_2017/businessletter_image2d.jpg)

we can identify the different letter parts such as the sender, date, recipient, greeting line, letter body, signature, and enclosures. These parts can then be displayed in an image:

![letter content](https://github.com/ReemHal/letter_images/blob/master/contours_letter_100.png)

The text content in each part can also be retrieved. Currently, the code uses the [Tesseract OCR tool](https://pypi.org/project/pytesseract/) to extract text content. With additional manipulation on the text and layout information, we can extract more data from the letter such as the name of the signator, the name of the sender and recipient and the purpose of the letter.

You can find more details in the __[letter_image_examples.ipynb](https://github.com/ReemHal/letter_images/blob/master/letter_image_examples.ipynb)__ notebook.

### Extract and Query information from a collection of letter images

We can also extract information from each image in a collection of letter images and save it in a csv file. Extracted information include: the location of each part in a letter, the part type (e.g. sender, signature, body, etc.), the text within that part, as well as the image name and full path. this information allows us to query our database for a veriety of purposes. Examples include:
  - retrieving all the letters that were signed by a given name
  - getting letters written on the same day
  - finding letters discussing similar topics.  
  
We can also query the database against a sample letter not in our collection to find letters in our collection that were sent to the same person as the sample letter, or letters that were signed by the same signator, or even to find letters with a similar layout as the sample letter.

More detailed examples can be found in the __[explore_letter_image_dataset.ipynb](https://github.com/ReemHal/letter_images/blob/master/explore_letter_image_dataset.ipynb)__




