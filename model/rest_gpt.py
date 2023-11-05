import time
import re
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import os
from langchain.callbacks.base import BaseCallbackManager
from langchain.chains.base import Chain
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.llms.base import BaseLLM
from langchain import OpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.requests import RequestsWrapper
from langchain.chains.llm import LLMChain
from .planner import Planner
from .api_selector import APISelector
from .caller import Caller
from utils import ReducedOpenAPISpec
from .teamleader import TeamLeader
from langchain.chat_models import ChatOpenAI

logger = logging.getLogger(__name__)
FORMULATE_ANSWER_PROMPT = """

Task: Given a question and the answer given by the user, your task is to combine them to be a declarative sentence

Example: 
Question: Who is your father? Please give me the name of your father for further proceeding?
User: Carlos
Expected output: The user's father is Carlos

{question}
{answer}


""" 
class RestGPT(Chain):
    """Consists of an agent using tools."""

    llm: BaseLLM
    api_spec: ReducedOpenAPISpec
    team_leader: TeamLeader
    planner: Planner
    api_selector: APISelector
    scenario: str = "tmdb"
    requests_wrapper: RequestsWrapper
    simple_parser: bool = False
    return_intermediate_steps: bool = False
    max_iterations: Optional[int] = 15
    max_execution_time: Optional[float] = None
    early_stopping_method: str = "force"

    def __init__(
        self,
        llm: BaseLLM,
        api_spec: ReducedOpenAPISpec,
        scenario: str,
        requests_wrapper: RequestsWrapper,
        caller_doc_with_response: bool = False,
        parser_with_example: bool = False,
        simple_parser: bool = False,
        callback_manager: Optional[BaseCallbackManager] = None,
        **kwargs: Any,
    ) -> None:
        if scenario in ['TMDB', 'Tmdb']:
            scenario = 'tmdb'
        if scenario in ['Spotify']:
            scenario = 'spotify' 
        if scenario not in ['tmdb', 'spotify']:
            raise ValueError(f"Invalid scenario {scenario}")
        teamleader_llm = OpenAI(openai_api_key="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", model_name="text-davinci-003", temperature=0.0, max_tokens=700)
        team_leader = TeamLeader(llm=teamleader_llm, scenario=scenario)

        planner_llm = OpenAI(openai_api_key="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", model_name="text-davinci-003", temperature=0.0, max_tokens=700)
        planner = Planner(llm=planner_llm, scenario=scenario)
        api_selector_llm = OpenAI(openai_api_key="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", model_name="text-davinci-003", temperature=0.0, max_tokens=700)
        api_selector = APISelector(llm=api_selector_llm, scenario=scenario, api_spec=api_spec)

        super().__init__(
            llm=llm, api_spec=api_spec, team_leader=team_leader, planner=planner, api_selector=api_selector, scenario=scenario,
            requests_wrapper=requests_wrapper, simple_parser=simple_parser, callback_manager=callback_manager, **kwargs
        )

    def save(self, file_path: Union[Path, str]) -> None:
        """Raise error - saving not supported for Agent Executors."""
        raise ValueError(
            "Saving not supported for RestGPT. "
            "If you are trying to save the agent, please use the "
            "`.save_agent(...)`"
        )

    @property
    def _chain_type(self) -> str:
        return "RestGPT"

    @property
    def input_keys(self) -> List[str]:
        """Return the input keys.

        :meta private:
        """
        return ["query"]

    @property
    def output_keys(self) -> List[str]:
        """Return the singular output key.

        :meta private:
        """
        return self.planner.output_keys
    
    def debug_input(self) -> str:
        print("Debug...")
        return input()

    def _should_continue(self, iterations: int, time_elapsed: float) -> bool:
        if self.max_iterations is not None and iterations >= self.max_iterations:
            return False
        if (
            self.max_execution_time is not None
            and time_elapsed >= self.max_execution_time
        ):
            return False

        return True

    def _return(self, output, intermediate_steps: list) -> Dict[str, Any]:
        self.callback_manager.on_agent_finish(
            output, color="green", verbose=self.verbose
        )
        final_output = output.return_values
        if self.return_intermediate_steps:
            final_output["intermediate_steps"] = intermediate_steps
        return final_output

    def _get_api_selector_background(self, planner_history: List[Tuple[str, str]]) -> str:
        if len(planner_history) == 0:
            return "No background"
        return "\n".join([step[1] for step in planner_history])

    def _should_continue_plan(self, plan) -> bool:
        if re.search("Continue", plan):
            return True
        return False
    
    def _should_end(self, plan) -> bool:
        if re.search("Final Answer", plan):
            return True
        return False
        
    def _handle_exception(self, global_history):
        formulate_answer_prompt = PromptTemplate(
            template=FORMULATE_ANSWER_PROMPT,
            input_variables=["question", "answer"],
        )

        formulate_answer_llm = OpenAI(openai_api_key="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", model_name="text-davinci-003", temperature=0.0, max_tokens=700)
        formulate_answer_chain = LLMChain(llm=formulate_answer_llm, prompt=formulate_answer_prompt)
        leader_response = self.team_leader.run(history=global_history)
        if "Question:" in leader_response:
            global_history.append(leader_response)
            answer = input(leader_response + "\n")
        
            answer = "User: " + answer
            answer = formulate_answer_chain.run(question=leader_response, answer=answer)
            global_history.append("User: " + answer)
            leader_response = answer
        else:
            leader_response = re.sub(r"Refutation:", "User:", leader_response).strip()
            global_history.append(leader_response)
        return leader_response
        	

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        query = inputs['query']
        query = "User query: " + query
        planner_history: List[Tuple[str, str]] = []
        global_history = []
        iterations = 0
        time_elapsed = 0.0
        start_time = time.time()
        global_history.append(query)
	
        plan = self.planner.run(input=query, history=planner_history, leader="")
        logger.info(f"Planner: {plan}")
        global_history.append("Planner: " + plan)
        while "I need user's clarification" in plan:
            leader_response = self._handle_exception(global_history=global_history)
            plan = self.planner.run(input=query, history=planner_history, leader=leader_response)
            logger.info(f"Planner: {plan}")
            global_history.append("Planner: " + plan)

        while self._should_continue(iterations, time_elapsed):
            tmp_planner_history = [plan]
            api_selector_history: List[Tuple[str, str, str]] = []
            api_selector_background = self._get_api_selector_background(planner_history)
       
            api_plan = self.api_selector.run(plan=plan, background=api_selector_background, leader="")
            global_history.append("API Selector: " + api_plan)
            while "I need user's clarification" in api_plan:
                leader_response = self._handle_exception(global_history=global_history)
                api_plan = self.api_selector.run(plan=plan, background=api_selector_background, leader=leader_response)
                global_history.append("API Selector: " + api_plan)

            finished = re.match(r"No API call needed.(.*)", api_plan)
            caller_llm = OpenAI(openai_api_key="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", model_name="text-davinci-003", temperature=0.0, max_tokens=700)
            if not finished:
                executor = Caller(llm=caller_llm, api_spec=self.api_spec, scenario=self.scenario, simple_parser=self.simple_parser, requests_wrapper=self.requests_wrapper)
                execution_res = executor.run(api_plan=api_plan, background=api_selector_background, leader="")
                global_history.append("Caller: " + execution_res)
                while "I need user's clarification" in execution_res:
                    leader_response = self._handle_exception(global_history=global_history)
                    execution_res = executor.run(api_plan=api_plan, background=api_selector_background, leader=leader_response)
                    global_history.append("Caller: " + execution_res)
            else:
                execution_res = finished.group(1)
                global_history.append("Caller: " + execution_res)
                
            global_history.append("End of Cycle")
            planner_history.append((plan, execution_res))
            api_selector_history.append((plan, api_plan, execution_res))
            
            objective = self.team_leader.run(history=global_history)
            global_history.append(objective)
            objective = re.sub(r"Objective to Planner:", "", objective).strip()
            if "Final Answer" in objective:
                logger.info(f"{objective}")
                return {"result": objective}
            logger.info(f"Objective: {objective}")
            
            plan = self.planner.run(input=objective, history=planner_history, leader="")
            logger.info(f"Planner: {plan}")
            global_history.append("Planner: " + plan)
            while "I need user's clarification" in plan:
                leader_response = self._handle_exception(global_history=global_history)
                plan = self.planner.run(input=query, history=planner_history, leader=leader_response)
                logger.info(f"Planner: {plan}")
                global_history.append("Planner: " + plan)

            while self._should_continue_plan(plan):
            
                api_selector_background = self._get_api_selector_background(planner_history)

                api_plan = self.api_selector.run(plan=tmp_planner_history[0], background=api_selector_background, history=api_selector_history, instruction=plan, leader="")
                global_history.append("API Selector: " + api_plan)
                while "I need user's clarification" in api_plan:
                    leader_response = self._handle_exception(global_history=global_history)
                    api_plan = self.api_selector.run(plan=plan, background=api_selector_background, leader=leader_response)
                    global_history.append("API Selector: " + api_plan)
                finished = re.match(r"No API call needed.(.*)", api_plan)

                caller_llm = OpenAI(openai_api_key="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", model_name="text-davinci-003", temperature=0.0, max_tokens=700)
                if not finished:
                    executor = Caller(llm=caller_llm, api_spec=self.api_spec, scenario=self.scenario, simple_parser=self.simple_parser, requests_wrapper=self.requests_wrapper)
                    execution_res = executor.run(api_plan=api_plan, background=api_selector_background, leader="")
                    global_history.append("Caller: " + execution_res)
                    while "I need user's clarification" in execution_res:
                        leader_response = self._handle_exception(global_history=global_history)
                        execution_res = executor.run(api_plan=api_plan, background=api_selector_background, leader=leader_response)
                        global_history.append("Caller: " + execution_res)
                else:
                    execution_res = finished.group(1)
                    global_history.append("Caller: " + execution_res)

                global_history.append("End of Cycle")
                planner_history.append((plan, execution_res))
                api_selector_history.append((plan, api_plan, execution_res))


                objective = self.team_leader.run(history=global_history)
                global_history.append(objective)
                objective = re.sub(r"Objective to Planner:", "", objective).strip()
                if "Final Answer" in objective:
                    logger.info(f"{objective}")
                    return {"result": objective}
                logger.info(f"Objective: {objective}")
                plan = self.planner.run(input=objective, history=planner_history, leader="")
                logger.info(f"Planner: {plan}")
                global_history.append("Planner: " + plan)
                while "I need user's clarification" in plan:
                    leader_response = self._handle_exception(global_history=global_history)
                    plan = self.planner.run(input=query, history=planner_history, leader=leader_response)
                    logger.info(f"Planner: {plan}")
                    global_history.append("Planner: " + plan)

            if self._should_end(plan):
                break

            iterations += 1
            time_elapsed = time.time() - start_time
            time.sleep(60)		
        return {"result": plan}
