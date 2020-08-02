'''
conda install -c conda-forge pyomo pyomo.extra
conda install -c conda-forge ipopt glpk
'''

'''
# <20.08.02> by KH
The issue is that if I run this optimization problem,

It is not working because of object function's data type problem
(It looks like object functions are int type now)

before I start coding something, I should check the data type of 'Param' and 'Set'

and also should check the how does it work

after then maybe I can handle the indexed constraints, variables

'''

from pyomo.environ import *
import pyomo.contrib.parmest.parmest as pamest
import pickle

class OPF_Optimization:
    def __init__(self, file_name):
        self.file_name = file_name
        self.bus_param = []
        self.line_param = []
        self.load_param = []
        self.gen_param = []
        self.cost_param= []
        
        self.bus_param_table = []
        self.line_param_table = []
        self.load_param_table = []
        self.gen_param_table = []
        self.cost_param_table = []

        self.buffer = []

    def get_network_model(self):
        with open(self.file_name,"rb") as fr:
            self.network = pickle.load(fr)

    def set_model(self):
        self.model = ConcreteModel()  

    def take_number(self):
        self.NUM_BUS = len(self.network['bus'])
        self.NUM_LINE = len(self.network['line'])
        self.NUM_GEN = len(self.network['gen'])
        self.NUM_LOAD = len(self.network['load'])

        self.NUM_BUS_PARAM = len(self.network['bus'].columns)
        self.NUM_LINE_PARAM = len(self.network['line'].columns)
        self.NUM_GEN_PARAM = len(self.network['gen'].columns)
        self.NUM_LOAD_PARAM = len(self.network['load'].columns)

    def set_bus_parameter(self):
        for bus_column in range(self.NUM_BUS_PARAM):
            for bus_nu in range(self.NUM_BUS):
                self.buffer.append(self.network['bus'][self.network['bus'].columns[bus_column]][bus_nu])

            self.bus_param.append(self.buffer.copy())
            self.buffer.clear()

            self.bus_param_table.append(self.network['bus'].columns[bus_column])

    def set_line_parameter(self):
        for line_column in range(self.NUM_LINE_PARAM):
            for line_nu in range(self.NUM_LINE):
                self.buffer.append(self.network['line'][self.network['line'].columns[line_column]][line_nu])

            self.line_param.append(self.buffer.copy())
            self.buffer.clear()

            self.line_param_table.append(self.network['line'].columns[line_column])

    def set_load_parameter(self):
        for load_column in range(self.NUM_LOAD_PARAM):
            for load_nu in range(self.NUM_LOAD):
                self.buffer.append(self.network['load'][self.network['load'].columns[load_column]][load_nu])

            self.load_param.append(self.buffer.copy())
            self.buffer.clear()

            self.load_param_table.append(self.network['load'].columns[load_column])

    def set_gen_parameter(self):
        for gen_column in range(self.NUM_GEN_PARAM):
            for gen_nu in range(self.NUM_GEN):
                self.buffer.append(self.network['gen'][self.network['gen'].columns[gen_column]][gen_nu])

            self.gen_param.append(self.buffer.copy())
            self.buffer.clear()

            self.gen_param_table.append(self.network['gen'].columns[gen_column])

    def set_cost_parameter(self):
        if len(self.network['pwl_cost'])==0:
            print("************************ loading poly cost function ************************")
            self.NUM_COST_PARAM = len(self.network['poly_cost'].columns)
            
            for cost_column in range(self.NUM_COST_PARAM):
                for gen_nu in range(self.NUM_GEN):
                    self.buffer.append(self.network['poly_cost'][self.network['poly_cost'].columns[cost_column]][gen_nu])
                
                self.cost_param.append(self.buffer.copy())
                self.buffer.clear()

                self.cost_param_table.append(self.network['poly_cost'].columns[cost_column])


    '''
    Optimization Part
    '''
    
    def set_variable(self):
        # find generation power index
        self.generation_power_index = self.gen_param_table.index('p_mw')
        self.maximum_generation_power_index = self.gen_param_table.index('max_p_mw')
        self.minimum_generation_power_index = self.gen_param_table.index('min_p_mw')
        self.model.generation_power = Var(range(self.NUM_GEN))

        # set and find generation cost thing
        self.model.generation_cost = Var(range(self.NUM_GEN))

        self.cp0_cost = self.cost_param_table.index('cp0_eur')
        self.cp1_cost = self.cost_param_table.index('cp1_eur_per_mw')
        self.cp2_cost = self.cost_param_table.index('cp2_eur_per_mw2')
        self.cq0_cost = self.cost_param_table.index('cq0_eur')
        self.cq1_cost = self.cost_param_table.index('cq1_eur_per_mvar')
        self.cq2_cost = self.cost_param_table.index('cq2_eur_per_mvar2')


    def set_constraints(self):
        self.model.limits = ConstraintList()
        for i in range(self.NUM_GEN):
            self.model.limits.add(self.model.generation_power[i]<=self.gen_param[self.maximum_generation_power_index][i])
            self.model.limits.add(self.model.generation_power[i]>=self.gen_param[self.minimum_generation_power_index][i])

            # Generation's Cost Calculation
            self.model.limits.add(self.model.generation_cost[i] == (self.cost_param[self.cp0_cost][i]) \
                                  + (self.cost_param[self.cp1_cost][i])*self.model.generation_power[i] \
                                  + (self.cost_param[self.cp2_cost][i])*(self.model.generation_power[i]**2))

    def set_object_function(self):
        self.model.obj = Objective( sum(self.model.generation_cost), sense=minimize)

    def set_solver(self):
        self.solver = SolverFactory("ipopt")
        # self.solver = SolverFactory("BONMIN")

    def solve(self):
        result = self.solver.solve(self.model, tee=True)
        return result


if __name__ == "__main__":
    test_model = OPF_Optimization("network_data")
    test_model.get_network_model()
    test_model.take_number()
    test_model.set_bus_parameter()
    test_model.set_line_parameter()
    test_model.set_load_parameter()
    test_model.set_gen_parameter()
    test_model.set_cost_parameter()
    
    test_model.set_model()
    test_model.set_variable()
    test_model.set_constraints()
    test_model.set_object_function()
    test_model.set_solver()
    test_model.solve
