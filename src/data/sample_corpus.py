"""
A small built-in corpus (30 documents, 5 topics) so the project runs with no
downloads.  Each document also carries a ``topic`` label that the benchmark
uses as ground truth for relevance.
"""

_RAW = [
    # ---- programming ----------------------------------------------------
    ("programming", "Python Programming Language",
     "Python is a popular programming language known for readable syntax. Software developers use Python for web development, automation and machine learning."),
    ("programming", "Machine Learning Basics",
     "Machine learning is a field of computer science where software learns patterns from data. Supervised learning trains models on labelled examples."),
    ("programming", "Computer Science Fundamentals",
     "Computer science studies algorithms, data structures and computation. Students learn how software and hardware cooperate to solve problems."),
    ("programming", "Computer Hardware Overview",
     "Computer hardware includes the processor, memory, storage and the motherboard. Hardware engineers design chips that run software efficiently."),
    ("programming", "Database Systems",
     "A database stores structured data and answers queries quickly. Indexes such as the inverted index speed up search in software systems."),
    ("programming", "Version Control with Git",
     "Git tracks changes to source code so that developers can collaborate. Branches let programmers experiment without breaking the main software."),
    # ---- space ------------------------------------------------------------
    ("space", "The Solar System",
     "The solar system contains the Sun, eight planets, moons and asteroids. Planets orbit the Sun because of gravity."),
    ("space", "Stars and Galaxies",
     "Stars are massive balls of burning gas. A galaxy such as the Milky Way holds billions of stars, planets and clouds of dust."),
    ("space", "Mars Exploration",
     "Rovers explore the surface of Mars searching for signs of ancient water. The red planet is a target for future human missions."),
    ("space", "Black Holes",
     "A black hole has gravity so strong that nothing, not even light, can escape. Astronomers detect black holes by watching stars orbit them."),
    ("space", "Space Telescopes",
     "Space telescopes observe distant galaxies without atmospheric distortion. Images of stars and nebulae help astronomers study the universe."),
    ("space", "Rocket Launches",
     "A rocket burns fuel to escape Earth's gravity and place satellites in orbit. Reusable rockets make space missions cheaper."),
    # ---- cooking ----------------------------------------------------------
    ("cooking", "Baking Bread at Home",
     "To bake bread, mix flour, water, yeast and salt, then let the dough rise. Bake the loaf in a hot oven until the crust turns golden."),
    ("cooking", "Italian Pasta Recipes",
     "Cook pasta in salted boiling water until firm. Toss the pasta with tomato sauce, garlic, olive oil and fresh basil."),
    ("cooking", "Grilling Vegetables",
     "Brush vegetables with oil and grill them until tender. Peppers, zucchini and onions taste smoky and sweet after grilling."),
    ("cooking", "Making Chocolate Cake",
     "Combine flour, sugar, cocoa and eggs to make cake batter. Bake the chocolate cake in the oven and let it cool before frosting."),
    ("cooking", "Spices in Indian Cooking",
     "Indian cooking uses spices such as turmeric, cumin and coriander. Toasting spices in hot oil releases their aroma before adding vegetables."),
    ("cooking", "Fermentation and Pickles",
     "Pickles are made by soaking cucumbers in brine. Fermentation produces tangy flavour and preserves vegetables without refrigeration."),
    # ---- sports -----------------------------------------------------------
    ("sports", "Football World Cup",
     "The football World Cup brings national teams together every four years. Strikers score goals while the goalkeeper defends the net."),
    ("sports", "Basketball Fundamentals",
     "Basketball players dribble, pass and shoot the ball into a hoop. A team wins by scoring more points than the opponent."),
    ("sports", "Cricket Match Formats",
     "Cricket matches are played as tests, one day games and twenty overs games. Batsmen score runs while bowlers try to take wickets."),
    ("sports", "Running a Marathon",
     "A marathon is a long distance race of forty two kilometres. Runners train for months to build stamina and avoid injury."),
    ("sports", "Tennis Tournaments",
     "Tennis players compete in tournaments on grass, clay and hard courts. A match is won by taking the most sets."),
    ("sports", "Swimming Techniques",
     "Swimmers use freestyle, backstroke, breaststroke and butterfly. Good technique reduces drag and improves speed in the pool."),
    # ---- health -----------------------------------------------------------
    ("health", "Healthy Diet Basics",
     "A healthy diet includes vegetables, fruit, whole grains and protein. Doctors recommend limiting sugar and processed food."),
    ("health", "How Vaccines Work",
     "A vaccine trains the immune system to recognise a virus. Doctors give vaccines to protect patients from serious disease."),
    ("health", "Sleep and Wellbeing",
     "Adults need seven to nine hours of sleep. Poor sleep harms memory, mood and the immune system."),
    ("health", "Exercise and Heart Health",
     "Regular exercise strengthens the heart and lowers blood pressure. Doctors advise thirty minutes of activity on most days."),
    ("health", "Common Medicines",
     "Pharmacists dispense medicine prescribed by a doctor. Antibiotics treat bacterial infections but do not work against a virus."),
    ("health", "Mental Health Care",
     "Therapy and medicine help patients manage anxiety and depression. Talking to a doctor or counsellor is an important first step."),
]

# doc ids start at 1
SAMPLE_DOCS = [
    {"id": i, "title": title, "text": text, "topic": topic}
    for i, (topic, title, text) in enumerate(_RAW, start=1)
]
