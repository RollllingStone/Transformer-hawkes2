import numpy as np
import torch
import torch.utils.data

from transformer import Constants


class EventData(torch.utils.data.Dataset):
    """ Event stream dataset. """

    def __init__(self, all_store_events):
        """
        Data should be a list of event streams; each event stream is a list of dictionaries;
        each dictionary contains: time_since_start, time_since_last_event, type_event
        {time since start(2019-01-01), time since last event, event demand, store index }
        Data is a list of stores, each store is a list of dictionaries representing events for this particular store
        inst = event stream 
        elem = dictionary of an event

        换成 data is a dict of list of dict
        {'placekey1': [event sequence of placekey 1], 'plackey2': [{{time since start(2019-01-01), time since last event, event demand}, ]}

        event data最后仍然给出同样的四个list sequence
        """

        self.time = [[single_event['time_since_start'] for single_event in single_store_events] for single_store_events in all_store_events]
        self.time_gap = [[single_event['time_since_last_event'] for single_event in single_store_events] for single_store_events in all_store_events]
        # plus 1 since there could be event type 0, but we use 0 as padding
        self.demand_marker = [[single_event['type_event'] + 0.00001 for single_event in single_store_events] for single_store_events in all_store_events]
        # plus 1 since there could be event type 0, but we use 0 as padding
        self.store_index = [[single_event['store_index'] + 1 for single_event in single_store_events] for single_store_events in all_store_events]
        self.event_type = [[1 for single_event in single_store_events] for single_store_events in all_store_events]

        self.length = len(all_store_events)


        # self.time = [[elem['time_since_start'] for elem in inst] for inst in data]
        # self.time_gap = [[elem['time_since_last_event'] for elem in inst] for inst in data]
        # plus 1 since there could be event type 0, but we use 0 as padding
        # self.length = len(data)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """ Each returned element is a list, which represents an event stream """
        return self.time[idx], self.time_gap[idx], self.event_type[idx], self.demand_marker[idx], self.store_index[idx]


def pad_time(all_store_events):
    """ Pad the instance to the max seq length in batch. """

    max_len = max(len(single_store_events) for single_store_events in all_store_events)

    # single sided padding
    batch_seq = np.array([
        single_store_events + [Constants.PAD] * (max_len - len(single_store_events))
        for single_store_events in all_store_events])


    # max_len = max(len(inst) for inst in insts)

    # batch_seq = np.array([
    #     inst + [Constants.PAD] * (max_len - len(inst))
    #     for inst in insts])

    return torch.tensor(batch_seq, dtype=torch.float32)


def pad_type(insts):
    """ Pad the instance to the max seq length in batch. """

    max_len = max(len(inst) for inst in insts)

    batch_seq = np.array([
        inst + [Constants.PAD] * (max_len - len(inst))
        for inst in insts])

    return torch.tensor(batch_seq, dtype=torch.long)


def pad_store(insts):
    """ Pad the instance to the max seq length in batch. """

    max_len = max(len(inst) for inst in insts)

    batch_seq = np.array([
        inst + [Constants.PAD] * (max_len - len(inst))
        for inst in insts])

    return torch.tensor(batch_seq, dtype=torch.long)



def pad_demand_mark(all_store_events):
    """ Pad the instance to the max seq length in batch. """

    max_len = max(len(single_store_events) for single_store_events in all_store_events)

    # single sided padding
    batch_seq = np.array([
        single_store_events + [Constants.PAD] * (max_len - len(single_store_events))
        for single_store_events in all_store_events])

    # max_len = max(len(inst) for inst in insts)

    # batch_seq = np.array([
    #     inst + [Constants.PAD] * (max_len - len(inst))
    #     for inst in insts])

    return torch.tensor(batch_seq, dtype=torch.float32)


# def collate_fn(all_store_events):
#     """ Collate function, as required by PyTorch. """

#     time, time_gap, event_type, event_demand, store_index= list(zip(*all_store_events))
#     time = pad_time(time)
#     time_gap = pad_time(time_gap)
#     event_demand = pad_demand_mark(event_demand)
#     event_type = pad_type(event_type)
#     store_index = pad_store(store_index)
#     return time, time_gap, event_type, event_demand, store_index 


# def get_dataloader(data, batch_size, shuffle=True):
#     """ Prepare dataloader. """

#     ds = EventData(data)

#     dl = torch.utils.data.DataLoader(
#         ds,
#         num_workers=2,
#         batch_size=batch_size,
#         collate_fn=collate_fn,
#         shuffle=shuffle
#     )
#     return dl


def collate_fn(all_store_events):
    """ Collate function, as required by PyTorch. """

    time, time_gap, event_type, event_demand, store_index= list(zip(*all_store_events))
    time = pad_time(time)
    time_gap = pad_time(time_gap)
    event_demand = pad_demand_mark(event_demand)
    event_type = pad_type(event_type)
    store_index = pad_store(store_index)
    return time, time_gap, event_type, event_demand, store_index 


def get_dataloader(data, batch_size, shuffle=True):
    """ Prepare dataloader. """

    ds = EventData(data)
    
    dl = torch.utils.data.DataLoader(
        ds,
        num_workers=2,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle
    )
    return dl
