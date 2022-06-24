from torch.utils.data import Dataset, DataLoader
import torch, gc

class CustomTextDataset(Dataset):
	'''
	CustomTextDataset object
	'''
	def __init__(self, sentences_list, tokenizer):
		'''
		Params:
			self: instance of object
			sentences_list (list of str): list of sentences
			tokenizer (tokenizer object): tokenizer function
		'''
		self.sentences_list = sentences_list
		self.tokenizer = tokenizer
	
	def __len__(self):
		'''
		Params:
			self: instance of object
		Returns:
			number of corpus texts
		'''
		return len(self.sentences_list)

class CustomTextDatasetMultiClass(CustomTextDataset):
	'''
	CustomTextDataset object for MultiClass Classification
	'''
	def __init__(self, sentences_list, labels_list, tokenizer, return_tensors_type="pt", max_length=512):
		'''
		Params:
			self: instance of object
			sentences_list (list of str): list of sentences
			labels_list (list of int): list of encoded labels
			tokenizer (tokenizer object): tokenizer function
			return_tensors_type (str): ['pt', 'tf'], default='pt'
			max_length (int): max length for tokenizer input
		'''
		CustomTextDataset.__init__(self, sentences_list, tokenizer)

		self.max_length = max_length
		self.return_tensors_type = return_tensors_type
		self.labels_list = labels_list

	def __getitem__(self, idx):
		'''
		Params:
			self: instance of object
			idx (int): index of iteration
		Returns:
			input_ids (pt tensors): encoded text as tensors
			attn_masks (pt tensors): attention masks as tensors
		'''
		text = self.sentences_list[idx]
		label = self.labels_list[idx]
		
		encodings_dict = self.tokenizer(text, truncation=True, max_length=self.max_length, padding="max_length")
		input_ids = torch.tensor(encodings_dict['input_ids'])
		attn_masks = torch.tensor(encodings_dict['attention_mask'])

		output_to_model = {'attention_mask':attn_masks, 'input_ids':input_ids, 'label':label}
		return output_to_model
