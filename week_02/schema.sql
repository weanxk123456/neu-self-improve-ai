-- Generated from the SQLAlchemy schema in mdp.py.
PRAGMA foreign_keys=ON;


CREATE TABLE actions (
	id INTEGER NOT NULL, 
	name VARCHAR NOT NULL, 
	PRIMARY KEY (id), 
	UNIQUE (name)
)

;


CREATE TABLE states (
	id INTEGER NOT NULL, 
	name VARCHAR NOT NULL, 
	terminal BOOLEAN NOT NULL, 
	PRIMARY KEY (id), 
	UNIQUE (name)
)

;


CREATE TABLE policy_actions (
	iteration INTEGER NOT NULL, 
	state_id INTEGER NOT NULL, 
	action_id INTEGER NOT NULL, 
	probability FLOAT NOT NULL, 
	PRIMARY KEY (iteration, state_id, action_id), 
	CHECK (probability >= 0 AND probability <= 1), 
	FOREIGN KEY(state_id) REFERENCES states (id), 
	FOREIGN KEY(action_id) REFERENCES actions (id)
)

;


CREATE TABLE state_values (
	iteration INTEGER NOT NULL, 
	state_id INTEGER NOT NULL, 
	value FLOAT NOT NULL, 
	PRIMARY KEY (iteration, state_id), 
	FOREIGN KEY(state_id) REFERENCES states (id)
)

;


CREATE TABLE transitions (
	state_id INTEGER NOT NULL, 
	action_id INTEGER NOT NULL, 
	next_state_id INTEGER NOT NULL, 
	probability FLOAT NOT NULL, 
	reward FLOAT NOT NULL, 
	PRIMARY KEY (state_id, action_id, next_state_id), 
	CHECK (probability >= 0 AND probability <= 1), 
	FOREIGN KEY(state_id) REFERENCES states (id), 
	FOREIGN KEY(action_id) REFERENCES actions (id), 
	FOREIGN KEY(next_state_id) REFERENCES states (id)
)

;