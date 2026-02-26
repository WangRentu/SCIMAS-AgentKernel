from agentkernel_standalone.toolkit.models.api.openai import OpenAIProvider
from agentkernel_standalone.mas.system.components import Messager, Recorder, Timer
from agentkernel_standalone.mas.action.components import CommunicationComponent, OtherActionsComponent
from agentkernel_standalone.mas.agent.components import (
    ProfileComponent,
    PlanComponent,
    PerceiveComponent,
    ReflectComponent,
    InvokeComponent,
)
from examples.scimas_mvp.components.state_component import ScimasStateComponent
from agentkernel_standalone.mas.environment.components import (
    RelationComponent,
    SpaceComponent,
    get_or_create_component_class,
)
from examples.scimas_mvp.custom_controller import CustomController

from examples.scimas_mvp.plugins.agent.invoke.EasyInvokePlugin import EasyInvokePlugin
from examples.scimas_mvp.plugins.agent.perceive.EasyPerceivePlugin import EasyPerceivePlugin
from examples.scimas_mvp.plugins.agent.profile.EasyProfilePlugin import EasyProfilePlugin
from examples.scimas_mvp.plugins.agent.state.EasyStatePlugin import EasyStatePlugin
from examples.scimas_mvp.plugins.agent.plan.EasyPlanPlugin import EasyPlanPlugin
from examples.scimas_mvp.plugins.agent.reflect.EasyReflectPlugin import EasyReflectPlugin

from examples.scimas_mvp.plugins.action.communication.EasyCommunicationPlugin import EasyCommunicationPlugin
from examples.scimas_mvp.plugins.action.research.ResearchActionsPlugin import ResearchActionsPlugin
from examples.scimas_mvp.plugins.environment.relation.EasyRelationPlugin import EasyRelationPlugin
from examples.scimas_mvp.plugins.environment.space.EasySpacePlugin import EasySpacePlugin
from examples.scimas_mvp.plugins.environment.science.CausalWorldPlugin import CausalWorldPlugin

# Agent plugin and component registry

agent_plugin_calss_map = {
    "EasyPerceivePlugin": EasyPerceivePlugin,
    "EasyProfilePlugin": EasyProfilePlugin,
    "EasyStatePlugin": EasyStatePlugin,
    "EasyPlanPlugin": EasyPlanPlugin,
    "EasyInvokePlugin": EasyInvokePlugin,
    "EasyReflectPlugin": EasyReflectPlugin,
}

agent_component_class_map = {
    "profile": ProfileComponent,
    "state": ScimasStateComponent,
    "plan": PlanComponent,
    "perceive": PerceiveComponent,
    "reflect": ReflectComponent,
    "invoke": InvokeComponent,
}

# Action plugin and component registry
action_component_class_map = {
    "communication": CommunicationComponent,
    "otheractions": OtherActionsComponent,
}
action_plugin_class_map = {
    "EasyCommunicationPlugin": EasyCommunicationPlugin,
    "ResearchActionsPlugin": ResearchActionsPlugin,
}
# Model class
model_class_map = {
    "OpenAIProvider": OpenAIProvider,
}
# Environment plugin and component registry
environment_component_class_map = {
    "relation": RelationComponent,
    "space": SpaceComponent,
    "science": get_or_create_component_class("science"),
}
environment_plugin_class_map = {
    "EasyRelationPlugin": EasyRelationPlugin,
    "EasySpacePlugin": EasySpacePlugin,
    "CausalWorldPlugin": CausalWorldPlugin,
}

system_component_class_map = {
    "messager": Messager,
    "recorder": Recorder,
    "timer": Timer,
}

RESOURCES_MAPS = {
    "agent_components": agent_component_class_map,
    "agent_plugins": agent_plugin_calss_map,
    "action_components": action_component_class_map,
    "action_plugins": action_plugin_class_map,
    "environment_components": environment_component_class_map,
    "environment_plugins": environment_plugin_class_map,
    "system_components": system_component_class_map,
    "models": model_class_map,
    "controller": CustomController,
}
