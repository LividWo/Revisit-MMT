import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class CaptionImageRetriever(nn.Module):

    def __init__(self, config, *inputs, **kwargs):
        super().__init__()
        self.bert = kwargs['bert']
        args = kwargs['args']

        embeding_weights = np.load(args.image_embedding_file)
        self.img_vocab, self.img_dim = embeding_weights.shape
        # print(embeding_weights[0])
        embeddings_matrix = np.zeros((self.img_vocab + 1, self.img_dim))
        embeddings_matrix[1:] = embeding_weights

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_vecs = torch.FloatTensor(embeddings_matrix).to(self.device)  # N, 2048

        self.text_to_hidden = nn.Linear(config.hidden_size, args.feature_dim, bias=False)
        self.image_to_hidden = nn.Linear(self.img_dim, args.feature_dim, bias=False)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, caption_input_ids, caption_segment_ids, caption_input_masks, labels=None):
        caption_vec = self.bert(caption_input_ids, caption_input_masks, caption_segment_ids)[-1]  # B, bert_dim
        caption_vec = self.text_to_hidden(caption_vec)  # B, feature_dim
        # caption_vec = F.normalize(caption_vec, 2, -1)  # B, feature_dim

        image_vecs = self.image_to_hidden(self.image_vecs)   # 29001, feature_dim
        # image_vecs = F.normalize(image_vecs, 2, -1)

        caption_vec = caption_vec.unsqueeze(1)  # B, 1, feature_dim
        dot_product = torch.matmul(caption_vec, image_vecs.t())  # B, 29001
        dot_product.squeeze_(1)
        if labels is not None:
            # dot_product = F.log_softmax(dot_product, dim=-1)
            return self.loss_fn(dot_product, labels)
        else:
            return dot_product


class CaptionImageTransformedRetriever(nn.Module):

    def __init__(self, config, *inputs, **kwargs):
        super().__init__()
        self.bert = kwargs['bert']
        args = kwargs['args']

        embeding_weights = np.load(args.image_embedding_file)
        img_vocab, self.region, self.visual_dim = embeding_weights.shape
        self.visual_features = torch.FloatTensor(embeding_weights)
        
        embeddings_matrix = np.zeros((self.img_vocab + 1, self.img_dim))
        embeddings_matrix[1:] = embeding_weights

        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.image_vecs = torch.FloatTensor(embeddings_matrix).to(self.device)  # 29001, 2048

        self.text_to_hidden = nn.Linear(config.hidden_size, args.feature_dim, bias=False)
        self.image_to_hidden = nn.Linear(self.img_dim, args.feature_dim, bias=False)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, caption_input_ids, caption_segment_ids, caption_input_masks, labels=None):
        caption_vec = self.bert(caption_input_ids, caption_input_masks, caption_segment_ids)[-1]  # B, bert_dim
        caption_vec = self.text_to_hidden(caption_vec)  # B, feature_dim
        # caption_vec = F.normalize(caption_vec, 2, -1)  # B, feature_dim

        image_vecs = self.image_to_hidden(self.image_vecs)   # 29001, feature_dim
        # image_vecs = F.normalize(image_vecs, 2, -1)

        caption_vec = caption_vec.unsqueeze(1)  # B, 1, feature_dim
        dot_product = torch.matmul(caption_vec, image_vecs.t())  # B, 29001
        dot_product.squeeze_(1)
        if labels is not None:
            # dot_product = F.log_softmax(dot_product, dim=-1)
            return self.loss_fn(dot_product, labels)
        else:
            return dot_product
