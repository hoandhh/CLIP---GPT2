"""
    Module contains final Model and all pieces of it.
"""

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor, GPT2LMHeadModel, GPT2Tokenizer


class ImageEncoder(nn.Module):
    """
    Encodes image and returns it's embedding.
    """

    def __init__(self, model, device="cpu"):
        super(ImageEncoder, self).__init__()

        self.device = device

        self.preprocessor = CLIPProcessor.from_pretrained(model)
        self.model = CLIPModel.from_pretrained(model).vision_model.to(self.device)

    def forward(self, image):
        # only one image at a time
        image = self.preprocessor(images=image, return_tensors="pt").to(self.device)
        image_features = self.model(**image)

        print(image_features.pooler_output.shape)

        return image_features.pooler_output


class ImageTextEncoder(nn.Module):
    """
    Encodes both image and text, then concatenates their embeddings.
    """

    def __init__(self, model, device="cpu"):
        super(ImageTextEncoder, self).__init__()

        self.device = device

        # Load processor and CLIP model
        self.processor = CLIPProcessor.from_pretrained(model)
        self.model = CLIPModel.from_pretrained(model).to(self.device)

    def forward(self, image, text):
        """
        Encode image and text, then concatenate their embeddings.
        """
        image_inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        text_inputs = self.processor(text=text, return_tensors="pt").to(self.device)

        image_features = self.model.vision_model(
            **image_inputs
        ).pooler_output  # (batch_size, img_dim)
        text_features = self.model.text_model(
            **text_inputs
        ).pooler_output  # (batch_size, text_dim)

        combined_features = torch.cat(
            (image_features, text_features), dim=-1
        )  # (batch_size, img_dim + text_dim)

        return combined_features


class Mapping(nn.Module):
    """
    Maps image embedding to GPT-2 embedding.
    """

    def __init__(
        self,
        ep_len,
        num_layers,
        embed_size,
        n_heads,
        forward_expansion,
        dropout,
        device="cpu",
    ):
        super(Mapping, self).__init__()

        self.ep_len = ep_len
        self.embed_size = embed_size

        self.device = device

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_size,
                nhead=n_heads,
                dim_feedforward=embed_size * forward_expansion,
                dropout=dropout,
                batch_first=True,
                device=device,
            ),
            num_layers=num_layers,
        ).to(self.device)

        self.mapper = nn.Linear(embed_size, ep_len * embed_size).to(self.device)

        self.init_weights()

    def forward(self, img_embedded, train_mode=False):
        x = self.transformer_encoder(img_embedded)
        x = self.mapper(x)

        x = x.view(
            *(
                [-1, self.ep_len, self.embed_size]
                if train_mode
                else [self.ep_len, self.embed_size]
            )
        )  # for batched input

        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(m.bias)

            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


class TextDecoder(nn.Module):
    """
    Processes embedding into caption.
    """

    def __init__(self, model, device="cpu"):
        super(TextDecoder, self).__init__()

        self.device = device

        self.tokenizer = GPT2Tokenizer.from_pretrained(model)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = GPT2LMHeadModel.from_pretrained(model).to(self.device)
        self.vocab_size = self.model.config.vocab_size

    def forward(self, embedding, attention_mask=None):
        text_features = self.model(
            inputs_embeds=embedding, attention_mask=attention_mask
        )

        return text_features.logits


class Net(nn.Module):
    """
    Final Model class. Puts all pieces together and generates caption based on image.
    """

    def __init__(
        self,
        clip_model,
        text_model,
        ep_len,
        num_layers,
        n_heads,
        forward_expansion,
        dropout,
        max_len,
        device="cpu",
    ):
        super(Net, self).__init__()

        self.device = device
        self.ep_len = ep_len

        # Thay thế ImageEncoder bằng ImageTextEncoder
        self.ie = ImageTextEncoder(model=clip_model, device=device)

        embed_size = self.ie.model.config.hidden_size * 2  # Do nối 2 vector ảnh + text

        self.mp = Mapping(
            ep_len=self.ep_len,
            num_layers=num_layers,
            embed_size=embed_size,  # Dùng embed_size mới sau khi nối
            n_heads=n_heads,
            forward_expansion=forward_expansion,
            dropout=dropout,
            device=device,
        )

        self.td = TextDecoder(model=text_model, device=device)

        assert (
            embed_size == self.td.model.config.n_embd
        ), "Embedding size mismatch: Image+Text embedding must match GPT-2 embedding size"

        self.max_len = max_len
        self.criterion = nn.CrossEntropyLoss()

        self.freeze_layers()

    def freeze_layers(self):
        for p in [
            *list(self.ie.parameters()),
            *list(self.td.parameters())[14:-14],
        ]:  # freeze everything, except 1st and last transformer layer in Decoder
            p.requires_grad = False

    def forward(self, img, text, temperature=1.0):
        if temperature <= 0.0:
            temperature = 1.0
            print("Temperature must be positive. Setting it to 1.0")

        with torch.no_grad():
            img_text_embedded = self.ie(img, text)

            img_mapped = self.mp(img_text_embedded)

            sos_emb = self.td.model.transformer.wte(
                torch.tensor(self.td.tokenizer.bos_token_id).to(self.device)
            ).unsqueeze(0)

            start_emb = torch.cat([sos_emb, img_mapped], dim=0)

            tokens = []
            for _ in range(self.max_len):
                if len(tokens):
                    tok_emb = self.td.model.transformer.wte(
                        torch.tensor(tokens).to(self.device)
                    )
                    emb = torch.cat([start_emb, tok_emb], dim=0)
                else:
                    emb = start_emb

                pos_emb = self.td.model.transformer.wpe(
                    torch.arange(emb.shape[0]).to(self.device)
                )

                emb += pos_emb
                pred = self.td(emb)

                pred = torch.softmax(pred / temperature, dim=-1)
                _, pred = torch.max(pred, dim=1)

                last_token = pred[-1].item()
                tokens.append(last_token)

                if last_token == self.td.tokenizer.eos_token_id:
                    break

        decoded = self.td.tokenizer.decode(tokens[:-1])
        return decoded.strip().capitalize(), tokens

    def train_forward(self, img_text_emb, trg_cap, att_mask):
        x, x_mask = trg_cap[:, :-1], att_mask[:, :-1]
        y = trg_cap[:, 1:]

        img_mapped = self.mp(img_text_emb, train_mode=True)

        text_emb = self.td.model.transformer.wte(x)

        x = torch.concat([img_mapped, text_emb], dim=1)
        x_mask = torch.concat(
            [torch.ones(x_mask.shape[0], self.ep_len).to(self.device), x_mask], dim=1
        )

        pos_emb = self.td.model.transformer.wpe(
            torch.arange(x.shape[1]).to(self.td.device)
        ).expand_as(x)

        x += pos_emb

        res = self.td(x, attention_mask=x_mask)
        res = torch.softmax(res, dim=2)

        loss = self.criterion(
            res[:, self.ep_len :, :].reshape(-1, res.shape[-1]), y.reshape(-1)
        )

        return loss


if __name__ == "__main__":
    for clip, text in [
        ["openai/clip-vit-base-patch32", "gpt2"],
        ["openai/clip-vit-large-patch14", "gpt2-medium"],
    ]:
        model = Net(
            clip_model=clip,
            text_model=text,
            ep_len=3,
            num_layers=6,
            n_heads=16,
            forward_expansion=4,
            dropout=0.1,
            max_len=20,
        )

        model.eval()
        sample_image = torch.randn(3, 224, 224)  # Giả lập ảnh
        sample_text = "A man is playing guitar"  # Giả lập văn bản mô tả

        output_caption, tokens = model(sample_image, sample_text)
        print(output_caption)

        model.train()
        N = 10
        emb = model.td.model.config.n_embd
        length = 20

        loss = model.train_forward(
            torch.rand(N, emb),
            torch.randint(1, 50000, (N, length)),
            att_mask=torch.concat(
                [torch.ones(N, length - 3), torch.zeros(N, 3)], dim=1
            ),
        )
        print(loss)

        print(
            f"Total number of parameters: {sum(p.numel() for p in model.parameters())}"
        )
        print(
            f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
        )
