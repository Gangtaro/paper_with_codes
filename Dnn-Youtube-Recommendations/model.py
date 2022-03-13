import torch
import torch.nn as nn

class Dnn(nn.Module):
    """_summary_

    Args:
        layers (list): the number of nodes of input layer, hidden layer, output layer(user vector dimension)
        dropout_p (float): probability of Dropout layers
        activation (str, optional): Activation Function. Defaults to 'relu'.
    """
    def __init__(self, layers, dropout_p, activation='relu') -> None:
        super().__init__()

        # initialize Class attributes
        self.layers = layers # 각 레이어의 노드 개수를 리스트 형태로 저장
        self.n_layers = len(self.layers) - 1 # 아웃풋 레이어를 제외한, 레이어의 개수
        self.dropout = dropout_p # dropout 레이어의 확률
        self.activation = activation # 활성함수 레이어
        activations = {'sigmoid': nn.Sigmoid(), 'tanh':nn.Tanh(), 'relu':nn.ReLU(), 'leakyrelu':nn.LeakyReLU(), 'none':None}
        self.activation_function = activations[activation.lower()]

        # Define Layers
        dnn_modules = list()
        for i in range(self.n_layers):
            dnn_modules.append(nn.Dropout(p=self.dropout))
            input_size = self.layers[i]
            output_size = self.layers[i+1] 
            dnn_modules.append(nn.Linear(input_size, output_size))
            if self.activation_function is not None:
                dnn_modules.append(self.activation_function)

        self.dnn_layers = nn.Sequential(*dnn_modules)
        self.apply(self._init_weights)

    # initialize weights
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight.data) # When activation function is ReLU, Kaiming He initialization is probably best option
            if module.bias is not None:
                module.bias.data.fill_(0.0)
    
    def forward(self, input_feature):
        return self.dnn_layers(input_feature)



class CandidateGeneration(nn.Module):
    def __init__(self, n_items, n_tokens, n_locations, n_occupations, watch_emb_dim, search_emb_dim, loc_emb_dim, ocp_emb_dim, layers, dropout_p = 0.2) -> None:
        super().__init__()

        self.n_items = n_items
        self.n_tokens = n_tokens
        self.n_locations = n_locations
        self.n_occupations = n_occupations
        self.watch_emb_dim = watch_emb_dim
        self.search_emb_dim = search_emb_dim
        self.loc_emb_dim = loc_emb_dim
        self.ocp_emb_dim = ocp_emb_dim

        self.layers = layers
        self.dropout_p = dropout_p

        # TODO make example age -> Dataset
        # TODO watch movies -> Dataset
        # TODO search_tokens -> Datset

        # Define Layers
        # - embedding moduels
        self.watch_embedding = nn.Embedding(self.n_items, self.watch_emb_dim) 
        self.search_embedding = nn.Embedding(self.n_tokens, self.search_emb_dim) 
        self.location_embeding = nn.Embedding(self.n_locations, self.loc_emb_dim) 
        self.occupation_embeding = nn.Embedding(self.n_occupations, self.ocp_emb_dim) 

        # - Deep Neural Network
        self.Dnn = Dnn(self.layers, self.dropout_p, activation='relu')
        # self.predict_layer = nn.Linear(in_features=self.layers[-1], out_features= user_vector_dim) # TODO FILL HERE : user_vector_dim
        # self.Softmax = nn.Softmax()

        self.apply(self._init_weights)

    # initialize weights
    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight.data, mean=0.0, std=0.01)
        elif isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight.data) # When activation function is ReLU, Kaiming He initialization is probably best option
            if module.bias is not None:
                module.bias.data.fill_(0.0)


    def forward(self, input_feature):
        batch_size = input_feature.size()[0]

        user_id, watch_movies, search_tockens, location, gender, occupation = torch.split(input_feature, [1, 1, 1, ], -1)

        # TODO 각 watch_movies, search_tockens는 list 자료형으로 전달 될 건데, 하나의 batch 그룹에서 여러 user가 있을건데, 
        # 각각의 watch_movies, search_tockens의 자료 개수가 다를 수 있다. 이걸 계산해주는 모듈을 만들어줘야한다. 
        for i, movie_id in enumerate(watch_movies):
            if i == 0 : 
                watch_e = self.watch_embedding(movie_id)
            else : 
                watch_e += self.watch_embedding(movie_id)
        watch_e = []
        search_e = []
        location_e = self.location_embeding(location)
        occupation_e = self.occupation_embeding(occupation)
        
        input_feature= torch.cat([watch_e, search_e, location_e, occupation_e], dim=-1) # TODO fill in here : Concatenate all features
        user_vectors = self.DNN(input_feature)
        return user_vectors 