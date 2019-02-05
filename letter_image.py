import os
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
from PIL import Image
import cv2
from skimage import measure
from skimage.io import imread, imsave
from skimage.color import gray2rgb, rgb2gray
from skimage.filters import gaussian
from difflib import SequenceMatcher

import nltk
import scipy.ndimage as ndimage
from dateutil.parser import parse
import re
import string

pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# ordered parts list although some items can be in one of several places. this can be enforced/permitted throught the code
letter_parts=['heading', 'sender', 'date', 'recipient', 'subject', 'greeting', 'body', 'signature', 'enclosures']

letter_part_info={'greeting':
                    {'part_type': 'firstOrExact',
                     'part_options': ['dear', 'to whom it may concern', 'sir', 'madam']},
                  'signature':
                    {'part_type': 'firstOrExact',
                     'part_options':['sincerely', 'regards', 'thank you', 'respectfully', 'yours', 'best regards',\
                     'we thank you for your attention to this matter', 'very truly yours']},
                  'date':
                    {'part_type': 'parse',
                     'part_options': []},
                  'subject':
                    {'part_type': 'firstWord',
                     'part_options': ['re', 'response', 'subject', 'ref', 'sub']},
                  'enclosures':
                    {'part_type': 'firstWord',
                     'part_options': ['p.s', 'encl', 'enclosure', 'enclosures', 'notes', 'note', \
                                      'attachements', 'attachment']},}

letter_part_rules={'heading':
                        {'after':[],
                         'before':['date','greeting','recipient', 'sender',\
                                   'subject','body','signature', 'enclosures']},
                   'recipient':
                       {'after':['heading','sender', 'date'],
                        'before':['greeting','recipient',\
                                  'body','signature', 'enclosures']},
                   'body':
                       {'after':['greeting','recipient', 'sender', 'subject', 'date'],
                        'before':['signature', 'enclosures']},
                   'signature':
                       {'after':['greeting','recipient', 'sender', 'subject', 'date', 'body', 'signature'],
                        'before':['enclosures']}
                  }

class letter_image:
    """ This class handles letters saved as images.

    Functions in this class identify blocks of text in the image,
    extract text within each block, identify different parts of the
    letter as sender/header, recipient, date, subject, letter boday,
    signature, and notes/enclosures. There are also functions to
    display the contours in the image, display the image, print
    out the raw text within the image (without segmenting it into blocks),
    and display the text within each part of the processed letter.

    Inputs:
        * dataset_dir: the path to the dir containing the image.
        * image_name: the name of the image.

    Attributes:
        clip: a list of dictionary items where each item represents a single part of the letter and consists of:
            'contour': a list of contours. Each item in the list is an array of contour points.
            'text': a list of strings found in the letter image. Each string is the text contained in a contour from 'contour' list.
            'part': Any one of: sender/header, recipient, date, subject, letter boday,
                    signature, or  notes/enclosures.
    """

    def __init__(self, dataset_dir, image_name):
        """
        Initializes the class object.

        Inputs:
        * dataset_dir: the path to the dir containing the image.
        * image_name: the name of the image.
        """
        self.image_name = image_name
        self.letter_img = imread(os.path.join(dataset_dir, image_name))
        if len(self.letter_img.shape)>2:
            self.letter_img = self.letter_img[:,:,:3] #remove alpha channel
        # self.clip is a list where each item marks a region in the letter image (a contour),
        # the text within that region, and the letter part label for the region from letter_parts list
        self.clip = []
        self.letter_part_info = letter_part_info

    def letter_part_patterns(self, part):
        return self.letter_part_info[part]['part_options']

    def process_letter(self):
        """Identifies different regions in letter image and assigns part labels to each region
        """
        contours = self.get_contours()
        self.get_text_in_contour_regions(contours)
        self.get_letter_parts()

    def get_contours(self, sigma_val= 5, cont_val =0.95):
        """Finds contiguous segments in the letter

           Inputs:
             * sigma(optional): the standard deviation for the Gaussian kernel
             * cont_val(optional): level value along which to find contours in the image.
        """

        blurred_letter = gaussian(self.letter_img, sigma=sigma_val)
        contours = measure.find_contours(rgb2gray(blurred_letter), cont_val)

        return contours

    def display_letter(self):
        """
           Display the image of the letter
        """
        fig, ax=plt.subplots(figsize=(15,15))
        ax.imshow(self.letter_img, cmap='Greys_r')
        plt.tick_params(
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            left=False,
            right=False,
            labeltop=False,
            labelleft=False,
            labelright=False,
            labelbottom=False) # labels along the bottom edge are off
        plt.show()

    def display_contours(self, display=True, save=False, savedir=''):
        """
           Display the identified regions super-imposed on the letter
        """

        if (self.clip != []):
            fig, ax=plt.subplots(figsize=(15,15))
            ax.imshow(self.letter_img)
            label_list = []
            for n, curr_clip in enumerate(self.clip):
                same_color = None
                for contour in curr_clip['contour']:
                    if same_color is None:
                        item = ax.plot(contour[:, 1], contour[:, 0], linewidth=2, label=curr_clip['part'])
                        same_color = item[0].get_color()
                        label_list = label_list+[curr_clip['part']]
                    else:
                        item = ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color=same_color)

            if not([None] == list(set(label_list))):
                plt.legend(loc='lower right')
            else:
                print("Error: No parts identified!")

            plt.tick_params(
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                left=False,
                right=False,
                labeltop=False,
                labelleft=False,
                labelright=False,
                labelbottom=False) # labels along the bottom edge are off
            if display:
                plt.show()
            if save:
                plt.savefig(os.path.join(savedir,'contours_'+self.image_name))
            if not(display):
                plt.close()
        else:
            print("Please call the process_letter function first!")

    def display_letter_parts(self):
        """
           Print each part of the letter and its part label from the letter_parts list
        """
        if (self.clip != []):
            for i in np.arange(len(self.clip)):
                for curr_text in self.clip[i]['text']:
                    print("{}\n".format(curr_text))
                print("type:{}\n=============".format(self.clip[i]['part']))

        else:
            print("Please call the process_letter function first!")


    def get_letter_text(self):
        """
           Get the unlabeled and unsegmented letter text from the letter image
        """

        text = pytesseract.image_to_string(self.letter_img)

        return(text)

    #Is it a name?
    def _first_word_check(self, text_words, part):
        firstWord_check = False
        firstWord = text_words[0].strip(string.punctuation)
        for curr_word in  letter_part_info[part]['part_options']:
            # fuzzy string check in case of ocr error
            if SequenceMatcher(a=firstWord.lower(), b=curr_word).ratio()>0.75:
                firstWord_check = True

        return firstWord_check

    #Is it a greeting or a signature?
    def _first_or_exact_check_(self, text, text_words, part):
        greeting_or_sig= False
        potential_matches = set(list({text})+list({text_words[0]}))
        for curr_letter_word in potential_matches:
            for curr_word in letter_part_info[part]['part_options']:
                # fuzzy string check in case of ocr error
                if SequenceMatcher(a=curr_letter_word.lower(), b=curr_word).ratio()>0.75:
                    greeting_or_sig= True
        return greeting_or_sig

    # Is the current part being inspected between the greeting and signature?
    def _between_greeting_and_sig_(self, current_part_order, known_parts):
        before_sig = not('signature' in known_parts.keys())  or\
                    ('signature' in known_parts.keys()) and\
                    (known_parts['signature']>current_part_order)
        after_greeting = ('greeting' in known_parts.keys()) and\
                         (known_parts['greeting']<current_part_order)
        return before_sig and after_greeting

    # Is the current part being inspected after the signature
    def _after_sig_(self, current_part_order, known_parts):
        after_sig = ('signature' in known_parts.keys()) and\
                    (known_parts['signature']>current_part_order)
        return after_sig


    def find_patterned_parts(self, text, letter_part_info, letter_parts, known_parts={}):
        """
            Check parts that are easily identifiable through patterns in parts_with_patterns dict
        """

        curr_part = None
        found = False
        cont_loop= True
        curr_index= -1

        # remove leading, trailing, and multiple spaces
        text = re.sub(' +', ' ', text.strip())
        text = text.lower()
        text_words = text.split()

        parts_with_patterns = letter_part_info.keys()
        for part in parts_with_patterns:
            # Is it a date?
            if not(part in known_parts.keys()):
                if (letter_part_info[part]['part_type'] == 'parse'):
                    curr_part = part
                    curr_index = letter_parts.index(part)
                    found=True
                    try:
                        parse(text, fuzzy_with_tokens=False)
                    except ValueError:
                        curr_part = None
                        curr_index = -1
                        found=False
                # Is it a greeting, signature, enclosure, or name?
                elif (letter_part_info[part]['part_type'] == 'firstOrExact'):
                    clean_text = re.sub('[^A-Za-z0-9 ]', '', text)
                    if (self._first_or_exact_check_(clean_text, text_words, part)):
                        curr_part= part
                        if (part in letter_parts):
                            curr_index = letter_parts.index(part)
                        else:
                            #handles meta parts such as "name" which should be
                            #replaced by a real part name eventually
                            curr_index = None
                        found=True
                elif (letter_part_info[part]['part_type'] == 'firstWord'):
                    if (self._first_word_check(text_words,part)):
                        curr_part= part
                        if (part in letter_parts):
                            curr_index = letter_parts.index(part)
                        else:
                            #handles meta parts such as "name" which should be
                            #replaced by a real part name eventually
                            curr_index = None
                        found=True

                if found==True:
                    break

        return curr_part, curr_index

    def parts_with_relative_loc(self, letter_part_info, letter_parts, known_parts):
        """
            Infer body, sender, recipient, and enclosure location from parts known so far
        """

        for part in letter_parts:
            start_index = None
            last_index = None
            if part in letter_part_rules.keys():
                #find the last occurrence of the parts in 'after' section of letter_part_rules[part]
                max_index = None
                for after_p in letter_part_rules[part]['after']:
                    if ((after_p in known_parts.keys()) and
                        ((max_index is None) or
                         (known_parts[after_p]>max_index))):
                        max_index= known_parts[after_p]
                start_index= 0 if (max_index is None) else max_index+1

                #find the first occurrence of the parts in 'before' section of letter_part_rules[part]
                min_index = None
                for before_p in letter_part_rules[part]['before']:
                    if ((before_p in known_parts.keys()) and
                        ((min_index is None) or
                         (known_parts[before_p]<min_index))):
                        min_index= known_parts[before_p]
                last_index = len(self.clip)-1 if (min_index is None) else min_index-1

                # part falls between start_index and last_index
                for clip_num in np.arange(start_index, last_index+1):
                    self.clip[clip_num]['part'] = part

                # Update known_parts
                if not(start_index is None) and \
                   not(last_index is None):
                    known_parts[part]= last_index

    def combine_identical_parts(self):
        """
            Combine consecutive letter regions that were found to belong to the same part.

            (Contours sometimes cause letter parts to be divided into consecutive connected
            regions (e.g. each paragraph in the letter's body))
        """

        prev_clip=None
        new_clip = []
        new_index = -1

        for i in np.arange(len(self.clip)):

            if not(prev_clip is None) and \
               (self.clip[i]['part'] == prev_clip):
                new_clip[new_index]['contour']= new_clip[new_index]['contour']+ self.clip[i]['contour']
                new_clip[new_index]['text'] = new_clip[new_index]['text']+ self.clip[i]['text']
            else:
                new_index +=1
                prev_clip = self.clip[i]['part']
                new_clip.append({'contour':[], 'text':'', 'part':None})
                new_clip[new_index]['contour']= self.clip[i]['contour']
                new_clip[new_index]['text'] = self.clip[i]['text']
                new_clip[new_index]['part'] = self.clip[i]['part']

        self.clip = new_clip

        return new_clip

    def get_text_in_contour_regions(self, contours):
        """
           Get text within each component.

           Input:
               * contours: list of contours in letter image.
        """

        from skimage.draw import polygon_perimeter

        clip=[]
        mask=[]
        n=0
        known_parts={}
        #gray_img = rgb2gray(self.letter_img).astype(np.int8)
        last_known_part_index = -1
        for i, contour in enumerate(contours):
            # Ignore tiny contours and large contours spanning more than the perimeter of the image
            if ((len(contour)>100) and \
                (len(contour)< (self.letter_img.shape[0]+self.letter_img.shape[1])*2)):

                curr_clip_img = 255 * np.ones_like(self.letter_img)
                mask= np.zeros(curr_clip_img.shape[:2], np.uint8)
                mask[np.round(contour[:, 0]).astype('int'), np.round(contour[:, 1]).astype('int')] = 1

                # Fill in the hole created by the contour boundary
                mask= ndimage.binary_fill_holes(mask).astype(np.uint8)
                curr_clip_img[(mask==1)] = self.letter_img[(mask==1)]

                """(x, y) = np.where(mask == 1)
                (topx, topy) = (np.min(x), np.min(y))
                (bottomx, bottomy) = (np.max(x), np.max(y))
                curr_clip_img = self.letter_img.copy()
                curr_clip_img = curr_clip_img[topx:bottomx+1, topy:bottomy+1]"""

                min_X = np.min(contour[:, 0])
                max_X = np.max(contour[:, 0])
                min_Y = np.min(contour[:, 1])
                max_Y = np.max(contour[:, 1])
                r = [min_X, max_X, max_X, min_X, min_X ]
                c = [max_Y, max_Y, min_Y, min_Y, max_Y]
                rr, cc = polygon_perimeter(r, c, curr_clip_img.shape)
                curr_clip_img[rr, cc] = 1

                # Find the text content in the current clip
                text = pytesseract.image_to_string(curr_clip_img) #.astype(np.uint8))
                # Only keep clips containing recognized text
                if (len(text)>0):
                    clip.append({'contour':[], 'text':'', 'part':None})
                    clip[n]['contour']= [contour] # We can replace the contour with the containing box for more efficiency. To Do.
                    clip[n]['text'] = [text]
                    n += 1

        self.clip = clip

        return self.clip


    def get_letter_parts(self):
        """
           The main function that identifies all the parts of the letter given the text in each contour/region
        """

        known_parts={}
        last_known_part_index = -1
        for i, clip_item in enumerate(self.clip):
            text = "\n".join(self.clip[i]['text'])
            if (text != None):
                #Check parts that are easily identifiable through patterns in parts_with_patterns
                (part, last_known_part_index) = self.find_patterned_parts(text, letter_part_info,\
                                                                                       letter_parts, known_parts)

                if (part != None):
                    clip_item['part'] = part
                    known_parts[clip_item['part']]=i


        #infer body, sender, recipeint, and enclosure location from parts known so far
        self.parts_with_relative_loc(letter_part_info, letter_parts, known_parts)

        #Finally, combine consecutive letter parts of the same type
        new_clip= self.combine_identical_parts()
        self.clip = new_clip


    def tel_or_fax(text):
        """
            Check if the input text is a tel and/or fax line

            Assumes at most a single phone number and at most a single fax number on the line.

            output: None or a str.search object containing matches for phone number and/or fax.

        """

        # Allowing non word characters to accomodate incorrect character recognition
        number_sequence = "[0-9]*[^\w\)]*[0-9]+"
        phone_abrevs = "(tel|telephone|phone)"
        fax_abrevs = "(fax|facsimile)"
        phone_pattern = "(\("+number_sequence+"\)|("+number_sequence+"))\s*\-?\s*"+\
                        number_sequence+"([\-|\s]"+number_sequence+")*"

        # a tel number can be preceded by "tel" or it may just be a number
        tel_pattern = "("+phone_abrevs+"|("+phone_abrevs+"\W))?\s*(?P<tel>"+phone_pattern+")"
        # a fax number must start with the word "fax"
        fax_pattern = "("+fax_abrevs+")?\W?\s*(?P<fax>"+phone_pattern+")"

        tel_or_fax_re_string = "^\s*"+tel_pattern+"\s*("+fax_pattern+")?\W*$"
        tel_or_fax_re = re.compile(tel_or_fax_re_string)
        tel_or_fax_res = tel_or_fax_re.search(text)
        if tel_or_fax_res is None:
            # Check if it is a fax number only
            fax_re_pattern = "^\s*("+fax_pattern+")\W*$"
            tel_or_fax_re = re.compile(fax_re_pattern)
            tel_or_fax_res = tel_or_fax_re.search(text)

        return tel_or_fax_res
