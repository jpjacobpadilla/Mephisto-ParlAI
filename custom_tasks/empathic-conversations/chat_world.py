#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.crowdsourcing.utils.worlds import CrowdOnboardWorld, CrowdTaskWorld  # type: ignore
from parlai.core.worlds import validate  # type: ignore
from parlai.agents._custom.remote import RemoteAgent

from joblib import Parallel, delayed  # type: ignore

from sqlalchemy import create_engine, text
from sqlalchemy.engine.base import Engine
import database  # Database credentials

from typing import List, Union


class MultiAgentDialogOnboardWorld(CrowdOnboardWorld):
    def __init__(self, opt, agent):
        super().__init__(opt, agent)
        self.opt = opt

    def parley(self):
        self.agent.agent_id = "Onboarding Agent"
        self.agent.observe({"id": "System", "text": "Welcome onboard!"})
        x = self.agent.act(timeout=self.opt["turn_timeout"])
        self.agent.observe(
            {
                "id": "System",
                "text": "Thank you for your input! Please wait while "
                "we match you with another worker...",
                "episode_done": True,
            }
        )
        self.episodeDone = True


class MultiAgentDialogWorld(CrowdTaskWorld):
    """
    Basic world where each agent gets a turn in a round-robin fashion, receiving as
    input the actions of all other agents since that agent last acted.
    """

    def __init__(self, opt, agents=None, shared=None):
        # Add passed in agents directly.
        self.agents = agents
        self.acts = [None] * len(agents)
        self.episodeDone = False
        self.max_turns = opt.get("max_turns", 5)
        self.current_turns = 0
        self.send_task_data = opt.get("send_task_data", False)
        self.opt = opt
        for idx, agent in enumerate(self.agents):
            agent.agent_id = f"Chat Agent {idx + 1}"

    def parley(self):
        """
        For each agent, get an observation of the last action each of the other agents
        took.
        Then take an action yourself.
        """
        acts = self.acts
        self.current_turns += 1
        for index, agent in enumerate(self.agents):
            try:
                acts[index] = agent.act(timeout=self.opt["turn_timeout"])
                if self.send_task_data:
                    acts[index].force_set(
                        "task_data",
                        {
                            "last_acting_agent": agent.agent_id,
                            "current_dialogue_turn": self.current_turns,
                            "utterance_count": self.current_turns + index,
                        },
                    )
            except TypeError:
                acts[index] = agent.act()  # not MTurkAgent
            if acts[index]["episode_done"]:
                self.episodeDone = True
            for other_agent in self.agents:
                if other_agent != agent:
                    other_agent.observe(validate(acts[index]))
        if self.current_turns >= self.max_turns:
            self.episodeDone = True
            for agent in self.agents:
                if isinstance(agent, RemoteAgent):
                    agent.observe({"text": "[DONE]"})
                    _ = agent.act()
                    continue
                agent.observe(
                    {
                    "id": "Coordinator",
                    "text": 
                        f'''    
                        You are done with the conversation. 
                        Please enter the following code into the below 
                        input box and continue with the rest of the survey.
                        
                        CODE: {agent.mephisto_agent.unit_id}
                        '''
                    }
                )

    def prep_save_data(self, agent):
        """Process and return any additional data from this world you may want to store"""
        return {"example_key": "example_value"}

    def episode_done(self):
        return self.episodeDone

    def shutdown(self):
        """
        Shutdown all mturk agents in parallel, otherwise if one mturk agent is
        disconnected then it could prevent other mturk agents from completing.
        """
        global shutdown_agent

        def shutdown_agent(agent):
            try:
                agent.shutdown(timeout=None)
            except Exception:
                agent.shutdown()  # not MTurkAgent

        Parallel(n_jobs=len(self.agents), backend="threading")(
            delayed(shutdown_agent)(agent) for agent in self.agents
        )


def make_onboarding_world(opt, agent):
    return MultiAgentDialogOnboardWorld(opt, agent)


def validate_onboarding(data):
    """Check the contents of the data to ensure they are valid"""
    print(f"Validating onboarding data {data}")
    return True


def get_world_params():
    return {"agent_count": 2}


def make_world(opt, agents):
    # The four following variables are currently hard coded and need to be manually changed.
    HOST_BOT = 'localhost'
    PORT_BOT = '35496'
    BOT_NAME = 'ECHO'
    LOCAL_SQLITE_PATH = '/Users/jacobpadilla/Desktop/Research/Mephisto/Mephisto-ParlAI/mephisto-data/data/database.db'

    bots = []

    while len(agents) + len(bots) < 2:
        bot = RemoteAgent({"host_bot": HOST_BOT, 
                           "port_bot": PORT_BOT})
        
        # agent_name_for_db should match the agent description in the database.
        bot.agent_name_for_db = BOT_NAME

        # This is a hack to skip the OverWorld and TaskWorld by 
        # sending dummy messages. OverWorld accept any message
        # and TaskWorld accept only "begin" as the identifier.
        bot.observe({"text": "dummy"})
        _ = bot.act()
        bot.observe({"text": "begin"})
        _ = bot.act()

        bots.append(bot)

    # Combine agents with bots
    agents.extend(bots)

    # May want to remove pymysql dbapi for use on GCP.
    conn_string = 'mysql+pymysql://{user}:{password}@{host}:{port}/{db}?charset:{encoding}'.format(
        user=database.user, password=database.password, host=database.host, 
        port=database.port, db=database.schema, encoding = 'utf-8')
    ec2_engine = create_engine(conn_string)

    # Now we need to make sure that the agents are in the agent table in the ec_2 schema, research database
    add_agents_db(ec2_engine, LOCAL_SQLITE_PATH, agents)
    # Add a conversation to the "conversation" table
    conversation_id = add_conversation_db(ec2_engine)
    # Add two rows to conversation_agent table
    merge_conversation_agents_db(ec2_engine, agents, conversation_id)

    return MultiAgentDialogWorld(opt, agents)


def add_agents_db(ec2_engine: Engine, sqlite_path: str, 
                  agents: List[Union["Agent", RemoteAgent]]) -> None:
    for agent in agents:
        if isinstance(agent, RemoteAgent):
            id = agent_db_setup(ec2_engine, agent.agent_name_for_db, 'c')
            agent.ec2_agent_id = id

        else:
            # Get worker_id that was sent via query string. This is located in the Mephisto local db
            local_engine = create_engine(f'sqlite:///{sqlite_path}')

            query = text('''
                select worker_name
                from workers as w
                inner join agents as a on a.worker_id=w.worker_id
                where a.unit_id = :unit_id;''')

            with local_engine.begin() as conn:
                result = conn.execute(query, {'unit_id': agent.mephisto_agent.unit_id})
                worker_name = result.fetchone()[0]

            print(f'Worker Name: {worker_name} Unit ID: {agent.mephisto_agent.unit_id}')

            agent.agent_name_for_db = worker_name
            id = agent_db_setup(ec2_engine, agent.agent_name_for_db, 'w')
            agent.ec2_agent_id = id


def agent_db_setup(engine: Engine, name: str, type: Union['w', 'c'], remove_sandbox_postfix=True) -> int:
    if remove_sandbox_postfix and name[-8:] == '_sandbox':
        name = name[:-8]

    query = text('''
        select agent_id
        from agent
        where description = :name;''')

    with engine.begin() as conn:
        result = conn.execute(query, {'name': name}).fetchone()
    
    if result is None:
        query = text('''
            insert into agent (agent_type, description) 
            values (:type, :name);''')
        
        with engine.begin() as conn:
            conn.execute(query, {'name': name, 'type': type})
            agent_id = conn.execute(text('select last_insert_id();')).fetchone()[0]

    else:
        agent_id = result[0]

    return agent_id


def add_conversation_db(ec2_engine: Engine) -> int:
    """
    This will add a new conversation to the table but will set article_id to NULL. 
    We do not know the exact article right now and will have to fill that out using the Qualtrics data.
    """
    query = text('''insert into conversation (article_id) values (NULL);''')
    
    with ec2_engine.begin() as conn:
        conn.execute(query)
        agent_id = conn.execute(text('select last_insert_id();')).fetchone()[0]

    return agent_id

def merge_conversation_agents_db(ec2_engine: Engine, agents: List[Union["Agent", RemoteAgent]], 
                                 conversation_id: int) -> None:
    query = text('''
        insert into conversation_agent (conversation_id, agent_id, agent_unit_id)
        values (:cid, :aid, :auid);''')

    for agent in agents:
        with ec2_engine.begin() as conn:
            params = {
                'cid': conversation_id,
                'aid': agent.ec2_agent_id,
                'auid': agent.mephisto_agent.unit_id if not isinstance(agent, RemoteAgent) else None
            }
            conn.execute(query, params)
