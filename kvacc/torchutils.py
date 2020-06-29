import unittest
import torch

from kvacc.commons import BaseTest

def collection_to(c, device):
    if torch.is_tensor(c):
        return c.to(device)
    else:
        if isinstance(c, dict):
            new_dict = {}
            for k, v in c.items():
                new_dict[k] = v.to(device) if torch.is_tensor(v) else torch.tensor(v).to(device)
            return new_dict
        elif isinstance(c, list):
            return list(map(lambda v: v.to(device) if torch.is_tensor(v) else torch.tensor(v).to(device), c))

        elif isinstance(c, tuple):
            return tuple(map(lambda v: v.to(device) if torch.is_tensor(v) else torch.tensor(v).to(device), c))
        elif isinstance(c, set):
            new_set = set()
            for v in c:
                new_set.add(v.to(device) if torch.is_tensor(v) else torch.tensor(v).to(device))
            return new_set
        else:
            raise ValueError('Input is not tensor and unknown collection type: %s' % type(c))

class TorchUtilsTest(BaseTest):
    def test_collection_to(self):
        dev = torch.device('cpu')

        self.assertTrue(collection_to(torch.tensor([1, 2, 3]), dev).device == dev)

        tc = collection_to({'A': [1, 2], 'B': [3]}, dev)
        print(tc)
        self.assertTrue(isinstance(tc, dict))
        self.assertTrue(torch.is_tensor(tc['A']))
        self.assertTrue(tc['A'].device == dev)

        tc = collection_to([1, 2, 3], dev)
        print(tc)
        self.assertTrue(isinstance(tc, list))
        self.assertTrue(torch.is_tensor(tc[1]))
        self.assertTrue(tc[1].device == dev)

        tc = collection_to((1, 2, 3), dev)
        print(tc)
        self.assertTrue(isinstance(tc, tuple))
        self.assertTrue(torch.is_tensor(tc[1]))
        self.assertTrue(tc[1].device == dev)

if __name__ == '__main__':
    unittest.main()
