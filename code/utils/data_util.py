label_list = ['no_relation', 'org:top_members/employees', 'org:members',
                'org:product', 'per:title', 'org:alternate_names',
                'per:employee_of', 'org:place_of_headquarters', 'per:product',
                'org:number_of_employees/members', 'per:children',
                'per:place_of_residence', 'per:alternate_names',
                'per:other_family', 'per:colleagues', 'per:origin', 'per:siblings',
                'per:spouse', 'org:founded', 'org:political/religious_affiliation',
                'org:member_of', 'per:parents', 'org:dissolved',
                'per:schools_attended', 'per:date_of_death', 'per:date_of_birth',
                'per:place_of_birth', 'per:place_of_death', 'org:founded_by',
                'per:religion']
                
ID_TO_LABEL = {i : label_list[i] for i in range(len(label_list))}
LABEL_TO_ID = {v: k for k, v in ID_TO_LABEL.items()}
MINOR_LABEL_IDS = [19, 21, 22, 24, 17, 20]
MINOR_LABELS = [ID_TO_LABEL[i] for i in MINOR_LABEL_IDS]

REF_SENT =  {20: 'The relation is "location of us county and county seat". location: the marking out of the boundaries, or identifying the place or site of, a piece of land, according to the description given in an entry, plan, map, etc. county: a region created by territorial division for the purpose of local government. county seat: the town or city that is the seat of government for a county.',
17: 'The relation is "people ethnicity and geographic distribution". ethnicity: an ethnic quality or affiliation resulting from racial or cultural ties. geographic: determined by geography. distribution: the spatial or geographic property of being scattered about over a range, area, or volume.',
24: 'The relation is "people deceased and place of burial". deceased: no longer alive, dead. place: an area, somewhere within an area. burial: the act of burying, interment, concealing something under the ground.',
21: 'The relation is "film and featured film locations". film: cinema; movies as a group. feature: to ascribe the greatest importance to something within a certain context. location: the marking out of the boundaries, or identifying the place or site of, a piece of land, according to the description given in an entry, plan, map, etc.',
19: 'The relation is "business company and advisor". business: commercial, industrial, or professional activity. company: any business, whether incorporated or not, that manufactures or sells products (also known as goods), or provides services as a commercial venture. advisor: one who offers advice, an expert who gives advice.',
22: 'The relation is "event and location". event: something that happens at a given place and time. location: the marking out of the boundaries, or identifying the place or site of, a piece of land, according to the description given in an entry, plan, map, etc.',}

