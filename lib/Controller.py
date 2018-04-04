import torch.nn as nn
from qrnn import QRNN
import FLAGS
from torch import Functional as F
from torch.distributions import Categorical
import torch

def uct_choice(
        self, curr_node): return curr_node["children"][curr_node["max_uct"]]

def select(self, root_node):
    curr_node = root_node

    while curr_node.children is not None:
        curr_node = self.uct_choice(curr_node)

    return curr_node

def U_func(P, N): return P/(1+N)
def Q_func(self, W, N): return W/N

def expand(self, curr_node, policy, value):
    curr_node["children"] = []
    curr_node["max_uct"] = {}

    max_U = 0
    max_U_idx = None

    for i, p in enumerate(policy):
        U = self.U_func(p, 0)

        child = {
            "N": 0,
            "W": 0,
            "Q": 0,
            "U": U,
            "P": p,
            "parent": curr_node,
            "idx": i
        }

        curr_node["children"].append(child)

        if U > max_U:
            max_U = U
            max_U_idx = i

    curr_node["max_uct"] = {
        "score": max_U, "idx": max_U_idx
    }

    return curr_node, value

def mcts_step(self, root_node):
    for _ in range(FLAGS.NUM_SIMS):
        leaf_node = self.select(root_node)
        #Hmmmmm so it needs to keep moving since it's auto regressive
        #So at each point we choose the decision based on the UCT
        #and we backup expansions

        expanded_node, value = self.expand(leaf_node, proba)

        root_node = self.backup(expanded_node, value)

    action, search_probas = self.choose_real_move(root_node)

    real_next_node = root_node["children"][action]
    del real_next_node["parent"]

    return real_next_node, action, search_probas

class Lookup(object):
    """Translates index lookups into attribute lookups."""

    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))


class Controller(nn.Module):
    def __init__(self):
        super(Controller, self).__init__()
        self.tanh = nn.Tanh
        self.sigmoid = nn.Sigmoid

        # I dont need to exactly clone this, I just want the general idea
        # If I can use stuff like QRNN's it might be more efficient anyways

        self.qrnn = QRNN(2*FLAGS.LSTM_SIZE, 4*FLAGS.LSTM_SIZE, dropout=.2)
        self.g_emb = QRNN(1, FLAGS.LSTM_SIZE, dropout=.2)

        # going to go kind of half way and make a bigger LSTM that is reused

        for _ in range(FLAGS.NUM_BRANCHES):
            self.add_module("start", QRNN(
                FLAGS.OUT_FILTERS, FLAGS.LSTM_SIZE, dropout=.2))
            self.add_module("count", QRNN(
                FLAGS.OUT_FILTERS, FLAGS.LSTM_SIZE, dropout=.2))

        self.start = Lookup(self, "start")
        self.count = Lookup(self, "count")

        self.start_critic = QRNN(FLAGS.LSTM_SIZE, 1, dropout=.2)
        self.count_critic = QRNN(FLAGS.LSTM_SIZE, 1, dropout=.2)

        self.w_attn_1 = QRNN(FLAGS.LSTM_SIZE, FLAGS.LSTM_SIZE, dropout=.2)
        self.w_attn_2 = QRNN(FLAGS.LSTM_SIZE, FLAGS.LSTM_SIZE, dropout=.2)
        self.v_attn = QRNN(FLAGS.LSTM_SIZE, 1, dropout=.2)

        # so lets see, ideally we want to keep the tree over time
        # so every time we run forward we make a new tree
        # and then after each decision point we throw away the top

    # soooo theres no doubt about this that it is really complicated
    # I need to basically just try it out like I did with MCTS and alphazero until I got it
    # just keep going through it, put what makes sense, and get the main idea
    # the main idea is sharing parameters.

    # sooo it is autoregressive where each decision is fed into the next input
    def forward(self, orig_inputs):
        # so every decision should have a vlaue and a probability
        arc_seq = []
        anchors = []
        anchors_w_1 = []

        inputs = self.g_emb(orig_inputs)

        root_node = {"children": None}
        for layer_id in range(FLAGS.NUM_LAYERS):
            for branch_id in range(FLAGS.NUM_BRANCHES):
                def do_branch(simulation):
                    # if this is autoregressive just continuing as normal
                    # will probably f it up. because it is imagining that it is continuing
                    # from where it was. although in theory it could maybe adapt to this
                    # and we are effectively building the simulations into the flow
                    # lets try it out.
                    def start_expansion():
                        start = dict()
                        start_qrnn = self.qrnn(inputs)
                        start_logits = self.start[branch_id](start_qrnn)

                        start_critic_input = start_qrnn.view(
                            inputs[0].size()[0], -1)

                        start["value"] = self.start_critic(start_critic_input)
                        start["probas"] = F.softmax(start_logits, axis=1)

                        return start

                    # I think I need to train the thing right here or it wont be able to
                    # remember what it did... I should look into passing around an embedding
                    # memory so I can easily save it, but its okay for now

                    #so let me see.. I'm selecting a decision
                    #then when I reach a leaf node I'm expanding it 
                    #and the expansion takes the probas and value to be valid
                    #the issue is I need to either re run the start qrnn, 
                    #or I need to somehow keep this moving, i.e. the num sims happens over
                    #time. I think that would work. I probably need a different root node
                    #for each decision, or maybe not
                    #so in expand I create num_actions worth of children
                    #I could change it to take the number of children

                    #so let me see.. I'm selecting an action here
                    #the issue is it can select multiple steps in the future
                    #Well in theory since this could probably repeat indefinitely 
                    #I think I can just have it select both the start and the count decisions
                    #and resume wherever it likes. I dont see an issue with that.

                    #Also I could probably reset the parameters for these to before the 
                    #simulations, but we'll have to see.

                    #If they are shared wont it be an issue if I reach a leaf node
                    #for one that isn't matching?
                    #I probably need to add a branch for that
                    #so if the leaf node is of type "start" expand is using the start 
                    #function

                    #okay so the idea is basically we start at the beginning
                    #we do alphazero select for all the actions
                    #depending on the leaf node type we decide what to expand with

                    leaf_node = self.select(root_node)

                    expanded_node = expand(leaf_node, start_probas, start_value)

                    #okay so I save the backup for the end, although, I 
                    #

                    root_node = self.backup(expanded_node, start_value)

                    for _ in range(FLAGS.NUM_SIMS):
                        leaf_node = self.select(root_node)
                        expanded_curr_node = expand(
                            curr_node, start_probas, start_value)

                    start_log_probas = F.log_softmax(start_logits, axis=1)
                    start_action = Categorical(start_log_probas)

                    if not simulation:
                        arc_seq.extend([start_action])

                    inputs = self.start[branch_id][start_action]

                    count_qrnn = self.qrnn(inputs)
                    count_logits = self.count[branch_id](count_qrnn)

                    # count_critic_input = count_qrnn.view(inputs[0].size()[0], -1)

                    # count_value = self.count_critic(count_critic_input)
                    # count_probas = F.softmax(count_logits, axis=1)

                    #action, search_probas = alphazero(count_probas, count_value)

                    count_log_probas = F.log_softmax(count_logits, axis=1)
                    count_action = Categorical(count_log_probas)

                    if not simulation:
                        arc_seq.extend([count_action + 1])

                    inputs = self.count[branch_id][count_action]
                for _ in range(FLAGS.NUM_SIMS):
                    do_branch(simulation=True)
                do_branch(simulation=False)

            # Branches done
            after_branches_qrnn = self.qrnn(inputs)

            if layer_id > 0:
                query = torch.stack(anchors_w_1)
                query = self.tanh(query + self.w_attn_2(after_branches_qrnn))

                query = self.v_attn(query)
                # size one
                skip_logit = torch.stack([-query, query])

                skip_proba = self.sigmoid(skip_logit)

                skip = Categorical(torch.log(skip_proba))
                arc_seq.append(skip)

                skip = skip.unsqueeze(0)
                inputs = skip.mm(torch.stack(anchors))
            else:
                # could be inputs or something
                inputs = self.g_emb(after_branches_qrnn)

            anchors.append(after_branches_qrnn)
            anchors_w_1.append(self.w_attn_1(after_branches_qrnn))

        # Layers Done
        arc_seq = torch.stack(arc_seq)

        return arc_seq

        # I dont really under stand it totally
        # it seems like it is getting a probability distribution
        # this is just a low level iterating through each of the placeholder variables
        # seems like she couldve maybe reused it, idk
        # but anyways what I effectively want is for it to produce one for each branch
