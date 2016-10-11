from pmst.source import Source


class Microscope:
    """A microscope""" 
    def __init__(self, source):
        if isinstance(source, Source):
            self.source = source
        else:
            raise ValueError('Argument must be a Source')
        self.component_list = []

    def add_component(self, component):
        self.component_list.append(component)

    def simulate(self):
        intersections = []
        for ray in self.source.rays:
            for component in self.component_list:
                intersections.append(component.intersection(ray))

        return intersections

    def plot_results():
        return 1
        
    def __str__(self):
        return 'Components:\t'+str(len(self.component_list))+'\n'


