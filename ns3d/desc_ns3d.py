#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Joy Hsu <joycj@stanford.edu>
#
# Distributed under terms of the MIT license.

from jacinle.utils.container import GView
from jacinle.config.environ_v2 import configs, set_configs
from datasets.program_translator import nscltree_to_nsclv2
from datasets.definition import gdef
from datasets.referit3d.codex_parsed_utterances import FunctionalTransformer, parse_codex_text, grammar

from models.model import NS3DReferIt3DModel
from models.utils.utils import canonize_monitors, update_from_loss_module
import pickle
import lark
import torch
import jactorch


with set_configs():
    configs.model.use_predefined_ccg = False
    configs.train.scene_add_supervision = True
    configs.train.refexp_add_supervision = True
    

class Model(NS3DReferIt3DModel):
    def __init__(self):
        super().__init__()
        
        codex_path = 'datasets/referit3d/data/codex_output.p'
        self.utterance_to_parsed_dict = pickle.load(open(codex_path, "rb"))
        self.parser = lark.Lark(grammar)
        self.trans = FunctionalTransformer()


    def forward(self, feed_dict):       
        feed_dict = GView(feed_dict)
        monitors, outputs = {}, {}
        
        f_sng = self.forward_sng(feed_dict)
        results = list()
        executions = list()
        parsings = list()
        referred_objs = list()
        reference_objs = list()
        
        exclusion_concept = list()
        sparsity_concept = list()
        symmetry_concept = list()
                
        for i in range(len(feed_dict.input_str)):
            this_size = feed_dict.input_objects_length[i]
            this_attribute = f_sng[i]['attribute'][:this_size, :]
            this_relation = f_sng[i]['relation'][:this_size, :this_size, :]
            this_multi_relation = f_sng[i]['multi_relation'][:this_size, :this_size, :]
            this_f_sng = {'attribute': this_attribute, 'relation': this_relation, 'multi_relation': this_multi_relation}

            with self.executor.with_grounding(self.grounding_cls(this_f_sng, self, self.training, feed_dict['input_objects_class'][i][:this_size], feed_dict['class_to_idx'][i])):
                
                parsing = parse_codex_text(self.parser, self.trans, self.utterance_to_parsed_dict[feed_dict.input_str[i]])
                program = parsing
                
                referred_objs.append(program.args[0].args[1].value)
                reference_objs.append(program.args[1].args[1].value)

                # exclusion
                if (len(program.args) == 3) and (program.args[2].value in self.excl_concept):
                    exclusion_concept.append(self.relation_embedding.similarity(this_relation, program.args[2].value))
                    
                # sparsity
                if len(program.args) == 3:
                    sparsity_concept.append(self.relation_embedding.similarity(this_relation, program.args[2].value))
                elif len(program.args) == 4:
                    feat = torch.cat([
                        jactorch.add_dim(this_multi_relation, 0, this_size),
                        jactorch.add_dim(this_multi_relation, 1, this_size),
                        jactorch.add_dim(this_multi_relation, 2, this_size)
                    ], dim=3)
                    sparsity_concept.append(self.multi_relation_embedding.similarity(feat, program.args[3].value))
                
                # symmetrical 
                if (len(program.args) == 3) and (program.args[2].value in self.sym_concept):
                    symmetry_concept.append(self.relation_embedding.similarity(this_relation, program.args[2].value))

                try:
                    execution = self.executor(program)
                except:
                    print('Errored: ' + feed_dict.input_str[i])
                    
            results.append((parsing, program, execution))
            executions.append(execution.value)
            parsings.append(parsing)  
        
        outputs['parsing'] = parsings
        outputs['results'] = results
        outputs['executions'] = executions
        outputs['referred_objs'] = referred_objs
        outputs['reference_objs'] = reference_objs
        update_from_loss_module(monitors, outputs, self.scene_concept_loss(feed_dict, f_sng, self.attribute_embedding, referred_objs, reference_objs))
        update_from_loss_module(monitors, outputs, self.refexp_loss(executions, feed_dict.output_target, feed_dict.input_objects_length))
        update_from_loss_module(monitors, outputs, self.regularization(excl_rel=exclusion_concept, sym_rel=symmetry_concept, spa_rel=sparsity_concept))

        if self.training:
            return monitors['loss/refexp'] + 100*monitors["loss/regular"], monitors, outputs
        else:
            outputs['monitors'] = monitors
            return outputs


def make_model(args):
    return Model()
