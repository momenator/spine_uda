"""Scheduler related functions"""

class LambdaLR():
    """lambda function class to decay learning rate to 0 after initial 
    epochs have passed
    """
    
    def __init__(self, init_lr, n_epochs, offset, decay_start_epoch):
        """Init functions
        Args:
            init_lr (float): initial learning rate
            n_epochs (int): number of epochs without decay
            offset (int): epoch offset - for continued training etc..
            decay_start_epoch (int): how many epochs to decay learning rate until 0
        
        """
        self.init_lr = init_lr
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

        
    def step(self, epoch):
        """Step function to decay learning rate
        
        Args:
            epoch (int): current epoch
            
        Returns:
            learning rate (float): decayed learning rate
        """
        
        epoch_tally = epoch + self.offset - self.n_epochs 
        decay_factor = max(0, epoch_tally) / float(self.decay_start_epoch)
        
        if decay_factor > 1:
            decay_factor = round(decay_factor)
                
        return self.init_lr - self.init_lr * decay_factor

