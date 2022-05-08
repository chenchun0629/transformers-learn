import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, AdamW


# 1、定义dataset
# 2、定义tokenizer, model
# 3、定义批处理方法
# 4、数据加载器
# 5、定义下游任务模型
# 6、训练
# 7、评估

# 1、定义dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, split):
        self.dataset = load_dataset(path='seamew/ChnSentiCorp', split=split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        text = self.dataset[i]['text']
        label = self.dataset[i]['label']

        return text, label


# 5、定义下游任务模型
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(768, 2)

        self.pretrained = AutoModel.from_pretrained('bert-base-chinese')

        # 不训练,不需要计算梯度
        for param in self.pretrained.parameters():
            param.requires_grad_(False)

    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            out = self.pretrained(input_ids=input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids)

        out = self.fc(out.last_hidden_state[:, 0])

        out = out.softmax(dim=1)

        return out


# 6、训练
def trainer(model, loader, device):
    model.to(device)
    # pretrained.to(device)
    # print(model)
    # 训练
    optimizer = AdamW(model.parameters(), lr=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(loader):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        token_type_ids = token_type_ids.to(device)
        labels = labels.to(device)
        # print(input_ids)
        # print(attention_mask)
        # print(token_type_ids)
        # print(labels)
        out = model(input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids)

        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % 5 == 0:
            out = out.argmax(dim=1)
            accuracy = (out == labels).sum().item() / len(labels)

            print(i, loss.item(), accuracy)

        if i == 300:
            break


# 7、评估
def test(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(loader):

        # if i == 5:
        #     break
        #
        # print(i)

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        token_type_ids = token_type_ids.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            out = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids)

        out = out.argmax(dim=1)
        correct += (out == labels).sum().item()
        total += len(labels)

    print(correct / total)


if __name__ == '__main__':
    dataset = Dataset('train')

    print(len(dataset), dataset[:10])

    # 2、定义tokenizer, model
    token = AutoTokenizer.from_pretrained('bert-base-chinese')

    # 3、定义批处理方法
    def collate_fn(data):
        sents = [i[0] for i in data]
        labels = [i[1] for i in data]

        # 编码
        data = token.batch_encode_plus(batch_text_or_text_pairs=sents,
                                       truncation=True,
                                       padding='max_length',
                                       max_length=500,
                                       return_tensors='pt',
                                       return_length=True)

        # input_ids:编码之后的数字
        # attention_mask:是补零的位置是0,其他位置是1
        input_ids = data['input_ids']
        attention_mask = data['attention_mask']
        token_type_ids = data['token_type_ids']
        labels = torch.LongTensor(labels)

        # print(data['length'], data['length'].max())

        return input_ids, attention_mask, token_type_ids, labels


    # 4、数据加载器
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=16,
                                         collate_fn=collate_fn,
                                         shuffle=True,
                                         drop_last=True)

    dev_loader = torch.utils.data.DataLoader(dataset=Dataset('validation'),
                                             batch_size=32,
                                             collate_fn=collate_fn,
                                             shuffle=True,
                                             drop_last=True)

    # 5、下游任务模型
    model = Model()

    # 6、训练
    device = "cuda:0"
    trainer(model, loader, device)

    # 7、评估
    test(model, dev_loader, device)
