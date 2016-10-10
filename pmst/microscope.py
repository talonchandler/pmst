class Microscope:
    """A microscope""" 
    def __init__(self, source):
        self.source = source
        self.component_list = []

    def add_component(self, component):
        self.component_list.append(component)

    def simulate():
        return 1

    def plot_results():
        return 1
        
    def __str__(self):
        return 'Components:\t'+str(len(self.component_list))+'\n'


