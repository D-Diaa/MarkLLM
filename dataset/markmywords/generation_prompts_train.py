import random

# Movie Review Topics
movie_review_topics = [
    ("Inception", "Christopher Nolan"),
    ("The Shawshank Redemption", "Frank Darabont"),
    ("Pulp Fiction", "Quentin Tarantino"),
    ("The Matrix", "The Wachowskis"),
    ("Forrest Gump", "Robert Zemeckis"),
    ("Parasite", "Bong Joon-ho"),
    ("The Godfather", "Francis Ford Coppola"),
    ("Schindler's List", "Steven Spielberg"),
    ("Spirited Away", "Hayao Miyazaki"),
    ("The Dark Knight", "Christopher Nolan"),
    ("Goodfellas", "Martin Scorsese"),
    ("City of God", "Fernando Meirelles"),
    ("The Silence of the Lambs", "Jonathan Demme"),
    ("Pan's Labyrinth", "Guillermo del Toro"),
    ("Whiplash", "Damien Chazelle"),
]

# Historical Event Summary Topics
historical_event_topics = [
    ("The Fall of the Berlin Wall", "1989"),
    ("The Moon Landing", "1969"),
    ("The French Revolution", "1789"),
    ("The Signing of the Magna Carta", "1215"),
    ("The Renaissance", "14th-17th century"),
    ("The Industrial Revolution", "18th-19th century"),
    ("The American Civil War", "1861-1865"),
    ("The Russian Revolution", "1917"),
    ("The Invention of the Printing Press", "c. 1440"),
    ("The Black Death", "1346-1353"),
    ("The Spanish Inquisition", "1478-1834"),
    ("The Discovery of the Americas", "1492"),
    ("The Atomic Bombings of Hiroshima and Nagasaki", "1945"),
    ("The Protestant Reformation", "16th century"),
    ("The Rise and Fall of the Roman Empire", "27 BC - 476 AD"),
]

# Tech Product Announcement Topics
tech_product_topics = [
    ("Quantum Computer", "QuantumTech"),
    ("Augmented Reality Glasses", "VisionX"),
    ("Self-Driving Electric Car", "AutoNova"),
    ("Brain-Computer Interface", "NeuraTech"),
    ("Fusion Reactor", "CleanEnergy Systems"),
    ("Holographic Display Smartphone", "HoloMobile"),
    ("AI-powered Personal Assistant Robot", "RoboComp"),
    ("Space Tourism Vessel", "GalacticVoyages"),
    ("Nanotechnology Medical Scanner", "NanoHealth"),
    ("Exoskeleton Suit", "AugmentCorp"),
    ("Artificial Photosynthesis Device", "GreenSynth"),
    ("Teleportation Device", "QuantumLeap Technologies"),
    ("Weather Control Satellite", "ClimateGuard"),
    ("Universal Language Translator Earbuds", "LinguaLink"),
    ("Artificial Organ Printer", "BioFab Innovations"),
]

# Biographical Sketch Topics
biographical_sketch_topics = [
    ("Marie Curie", "Physics and Chemistry"),
    ("Leonardo da Vinci", "Art and Science"),
    ("Nelson Mandela", "Politics and Civil Rights"),
    ("Ada Lovelace", "Mathematics and Computing"),
    ("Mahatma Gandhi", "Indian Independence Movement"),
    ("Frida Kahlo", "Art"),
    ("Albert Einstein", "Physics"),
    ("Jane Goodall", "Primatology and Anthropology"),
    ("William Shakespeare", "Literature"),
    ("Malala Yousafzai", "Education Activism"),
    ("Stephen Hawking", "Theoretical Physics"),
    ("Florence Nightingale", "Nursing"),
    ("Pablo Picasso", "Art"),
    ("Rosa Parks", "Civil Rights"),
    ("Charles Darwin", "Biology"),
]

# Environmental Impact Report Topics
environmental_impact_topics = [
    ("Deforestation", "Amazon Rainforest"),
    ("Plastic Pollution", "Pacific Ocean"),
    ("Oil Spill", "Gulf of Mexico"),
    ("Nuclear Disaster", "Chernobyl"),
    ("Coral Bleaching", "Great Barrier Reef"),
    ("Air Pollution", "Beijing"),
    ("Melting Ice Caps", "Arctic"),
    ("Desertification", "Sahel Region"),
    ("Water Scarcity", "Cape Town"),
    ("Overfishing", "North Atlantic"),
    ("Urban Sprawl", "Los Angeles"),
    ("Chemical Contamination", "Love Canal"),
    ("Loss of Biodiversity", "Madagascar"),
    ("E-waste Dumping", "Agbogbloshie, Ghana"),
    ("Fracking", "Pennsylvania"),
]

# Recipe Instruction Topics
recipe_instruction_topics = [
    ("Spaghetti Carbonara", "Italian"),
    ("Sushi Rolls", "Japanese"),
    ("Chicken Tikka Masala", "Indian"),
    ("Beef Bourguignon", "French"),
    ("Tacos al Pastor", "Mexican"),
    ("Pad Thai", "Thai"),
    ("Wiener Schnitzel", "Austrian"),
    ("Feijoada", "Brazilian"),
    ("Peking Duck", "Chinese"),
    ("Moussaka", "Greek"),
    ("Pho", "Vietnamese"),
    ("Bobotie", "South African"),
    ("Paella", "Spanish"),
    ("Pierogi", "Polish"),
    ("New England Clam Chowder", "American"),
]

# Travel Destination Guide Topics
travel_destination_topics = [
    ("Machu Picchu", "Peru"),
    ("Santorini", "Greece"),
    ("Kyoto", "Japan"),
    ("Serengeti National Park", "Tanzania"),
    ("Venice", "Italy"),
    ("Petra", "Jordan"),
    ("Banff National Park", "Canada"),
    ("Angkor Wat", "Cambodia"),
    ("Dubrovnik", "Croatia"),
    ("Cappadocia", "Turkey"),
    ("Galápagos Islands", "Ecuador"),
    ("Reykjavik", "Iceland"),
    ("Marrakech", "Morocco"),
    ("Great Barrier Reef", "Australia"),
    ("Cusco", "Peru"),
]

# Scientific Explanation Topics
scientific_explanation_topics = [
    ("Black Holes", "Astrophysics"),
    ("DNA Replication", "Molecular Biology"),
    ("Quantum Entanglement", "Quantum Physics"),
    ("Plate Tectonics", "Geology"),
    ("Photosynthesis", "Plant Biology"),
    ("Neural Networks", "Artificial Intelligence"),
    ("Climate Change", "Environmental Science"),
    ("Antibiotics Resistance", "Microbiology"),
    ("Dark Matter", "Cosmology"),
    ("Genetic Engineering", "Biotechnology"),
    ("Higgs Boson", "Particle Physics"),
    ("Stem Cells", "Regenerative Medicine"),
    ("Blockchain Technology", "Computer Science"),
    ("Evolution by Natural Selection", "Evolutionary Biology"),
    ("Gravitational Waves", "Physics"),
]

# Social Media Trend Analysis Topics
social_media_trend_topics = [
    ("#TikTokDance", "Video Sharing"),
    ("Instagram Reels", "Short-form Content"),
    ("Twitter Spaces", "Audio Chat"),
    ("NFTs on Social Media", "Digital Ownership"),
    ("BeReal App", "Authentic Moments"),
    ("LinkedIn Stories", "Professional Networking"),
    ("Facebook Groups", "Community Building"),
    ("YouTube Shorts", "Vertical Video"),
    ("Pinterest Idea Pins", "Creative Inspiration"),
    ("Snapchat Spotlight", "User-generated Content"),
    ("Reddit AMAs", "Q&A Sessions"),
    ("Clubhouse", "Drop-in Audio"),
    ("Instagram Shop", "Social Commerce"),
    ("TikTok For Business", "Brand Marketing"),
    ("Twitter Fleets", "Ephemeral Content"),
]

# Art Movement Analysis Topics
art_movement_topics = [
    ("Impressionism", "Late 19th century"),
    ("Cubism", "Early 20th century"),
    ("Surrealism", "1920s"),
    ("Pop Art", "1950s-1960s"),
    ("Abstract Expressionism", "1940s-1950s"),
    ("Baroque", "17th-18th centuries"),
    ("Renaissance", "14th-17th centuries"),
    ("Art Nouveau", "1890-1910"),
    ("Dadaism", "1915-1924"),
    ("Romanticism", "Late 18th-19th centuries"),
    ("Minimalism", "1960s-1970s"),
    ("Fauvism", "Early 20th century"),
    ("Bauhaus", "1919-1933"),
    ("Art Deco", "1920s-1930s"),
    ("Post-Impressionism", "1886-1905"),
]

# Psychological Disorder Explanation Topics
psychological_disorder_topics = [
    ("Major Depressive Disorder", "Mood Disorders"),
    ("Generalized Anxiety Disorder", "Anxiety Disorders"),
    ("Schizophrenia", "Psychotic Disorders"),
    ("Bipolar Disorder", "Mood Disorders"),
    ("Obsessive-Compulsive Disorder", "Anxiety Disorders"),
    ("Post-Traumatic Stress Disorder", "Trauma-Related Disorders"),
    ("Attention Deficit Hyperactivity Disorder", "Neurodevelopmental Disorders"),
    ("Autism Spectrum Disorder", "Neurodevelopmental Disorders"),
    ("Eating Disorders", "Feeding and Eating Disorders"),
    ("Borderline Personality Disorder", "Personality Disorders"),
    ("Social Anxiety Disorder", "Anxiety Disorders"),
    ("Dissociative Identity Disorder", "Dissociative Disorders"),
    ("Panic Disorder", "Anxiety Disorders"),
    ("Narcissistic Personality Disorder", "Personality Disorders"),
    ("Substance Use Disorder", "Addiction and Related Disorders"),
]

# Musical Genre Exploration Topics
musical_genre_topics = [
    ("Jazz", "20th century America"),
    ("Classical", "Western tradition"),
    ("Rock and Roll", "1950s America"),
    ("Hip Hop", "1970s New York"),
    ("Electronic Dance Music", "Late 20th century"),
    ("Blues", "Late 19th century America"),
    ("Country", "Southern United States"),
    ("Reggae", "1960s Jamaica"),
    ("Punk Rock", "1970s"),
    ("R&B", "1940s America"),
    ("Heavy Metal", "Late 1960s Britain"),
    ("Folk", "Traditional music"),
    ("Salsa", "1960s New York"),
    ("K-Pop", "South Korea"),
    ("Indie Rock", "1980s underground scene"),
]

# Philosophical Concept Explanation Topics
philosophical_concept_topics = [
    ("Existentialism", "Meaning and Existence"),
    ("Utilitarianism", "Ethics and Morality"),
    ("Dualism", "Mind-Body Problem"),
    ("Nihilism", "Meaning and Value"),
    ("Stoicism", "Ethics and Emotions"),
    ("Empiricism", "Theory of Knowledge"),
    ("Deontology", "Moral Philosophy"),
    ("Solipsism", "Nature of Reality"),
    ("Determinism", "Free Will"),
    ("Phenomenology", "Consciousness and Experience"),
    ("Pragmatism", "Truth and Meaning"),
    ("Relativism", "Truth and Morality"),
    ("Idealism", "Nature of Reality"),
    ("Skepticism", "Knowledge and Certainty"),
    ("Absurdism", "Meaning of Life"),
]

# Technological Innovation Explanation Topics
tech_innovation_topics = [
    ("5G Networks", "Telecommunications"),
    ("CRISPR Gene Editing", "Biotechnology"),
    ("Autonomous Vehicles", "Transportation"),
    ("Blockchain", "Cryptography and Finance"),
    ("Internet of Things", "Connected Devices"),
    ("Augmented Reality", "Human-Computer Interaction"),
    ("3D Printing", "Manufacturing"),
    ("Quantum Computing", "Information Processing"),
    ("Machine Learning", "Artificial Intelligence"),
    ("Renewable Energy", "Sustainability"),
    ("Cybersecurity", "Information Protection"),
    ("Cloud Computing", "Distributed Computing"),
    ("Nanotechnology", "Materials Science"),
    ("Virtual Reality", "Immersive Technology"),
    ("Robotics", "Automation and AI"),
]

# Literary Genre Analysis Topics
literary_genre_topics = [
    ("Gothic Fiction", "18th-19th centuries"),
    ("Magical Realism", "20th century"),
    ("Cyberpunk", "1980s"),
    ("Romantic Poetry", "Late 18th-19th centuries"),
    ("Modernist Literature", "Early 20th century"),
    ("Postcolonial Literature", "Post-World War II"),
    ("Beat Generation", "1950s-1960s"),
    ("Harlem Renaissance", "1920s-1930s"),
    ("Existentialist Literature", "Mid-20th century"),
    ("Dystopian Fiction", "20th-21st centuries"),
    ("Absurdist Drama", "Mid-20th century"),
    ("Southern Gothic", "20th century American South"),
    ("Transcendentalism", "19th century America"),
    ("Stream of Consciousness", "Early 20th century"),
    ("Naturalism", "Late 19th-early 20th centuries"),
]

# Architectural Style Exploration Topics
architectural_style_topics = [
    ("Gothic Architecture", "12th-16th centuries"),
    ("Bauhaus", "1919-1933"),
    ("Art Deco", "1920s-1930s"),
    ("Modernism", "20th century"),
    ("Brutalism", "1950s-1970s"),
    ("Postmodernism", "Late 20th century"),
    ("Islamic Architecture", "7th century onwards"),
    ("Neoclassicism", "18th-19th centuries"),
    ("Deconstructivism", "Late 20th century"),
    ("Victorian Architecture", "19th century"),
    ("Constructivism", "1920s-1930s Soviet Union"),
    ("Organic Architecture", "20th century"),
    ("High-Tech Architecture", "1970s onwards"),
    ("Baroque", "17th-18th centuries"),
    ("Futurism", "Early 20th century"),
]

# Economic Theory Analysis Topics
economic_theory_topics = [
    ("Keynesian Economics", "John Maynard Keynes"),
    ("Monetarism", "Milton Friedman"),
    ("Classical Economics", "Adam Smith"),
    ("Marxian Economics", "Karl Marx"),
    ("Austrian School", "Carl Menger"),
    ("Behavioral Economics", "Daniel Kahneman"),
    ("Supply-Side Economics", "Arthur Laffer"),
    ("New Institutional Economics", "Ronald Coase"),
    ("Post-Keynesian Economics", "Joan Robinson"),
    ("Neoclassical Economics", "Alfred Marshall"),
    ("Feminist Economics", "Barbara Bergmann"),
    ("Ecological Economics", "Herman Daly"),
    ("Game Theory", "John von Neumann"),
    ("New Trade Theory", "Paul Krugman"),
    ("Public Choice Theory", "James M. Buchanan"),
]

# Sports Event Summary Topics
sports_event_topics = [
    ("FIFA World Cup 2022", "Soccer"),
    ("Super Bowl LV", "American Football"),
    ("Wimbledon 2023", "Tennis"),
    ("Tokyo Olympics 2020", "Multi-sport"),
    ("Tour de France 2023", "Cycling"),
    ("NBA Finals 2023", "Basketball"),
    ("Cricket World Cup 2023", "Cricket"),
    ("Masters Tournament 2023", "Golf"),
    ("Kentucky Derby 2023", "Horse Racing"),
    ("Boston Marathon 2023", "Marathon Running"),
    ("Stanley Cup Finals 2023", "Ice Hockey"),
    ("Rugby World Cup 2023", "Rugby"),
    ("World Athletics Championships 2023", "Track and Field"),
    ("Formula 1 World Championship 2023", "Auto Racing"),
    ("UFC 300", "Mixed Martial Arts"),
]

# Language Learning Guide Topics
language_learning_topics = [
    ("Spanish", "Romance Languages"),
    ("Mandarin Chinese", "Sino-Tibetan Languages"),
    ("French", "Romance Languages"),
    ("Arabic", "Semitic Languages"),
    ("Japanese", "Japonic Languages"),
    ("German", "Germanic Languages"),
    ("Russian", "Slavic Languages"),
    ("Italian", "Romance Languages"),
    ("Korean", "Koreanic Languages"),
    ("Portuguese", "Romance Languages"),
    ("Hindi", "Indo-Aryan Languages"),
    ("Swedish", "Germanic Languages"),
    ("Turkish", "Turkic Languages"),
    ("Dutch", "Germanic Languages"),
    ("Greek", "Hellenic Languages"),
]

# Climate Phenomenon Explanation Topics
climate_phenomenon_topics = [
    ("El Niño", "Ocean-Atmosphere Interaction"),
    ("Greenhouse Effect", "Atmospheric Warming"),
    ("Polar Vortex", "Arctic Weather Pattern"),
    ("Monsoons", "Seasonal Wind Shifts"),
    ("Hurricanes", "Tropical Cyclones"),
    ("Jet Stream", "Atmospheric Currents"),
    ("Ocean Acidification", "Carbon Dioxide Absorption"),
    ("Ozone Depletion", "Stratospheric Ozone Loss"),
    ("Urban Heat Island", "Metropolitan Climate Change"),
    ("Permafrost Thawing", "Arctic Climate Change"),
    ("La Niña", "Pacific Ocean Cooling"),
    ("Albedo Effect", "Surface Reflectivity"),
    ("Desertification", "Land Degradation"),
    ("Sea Level Rise", "Coastal Climate Impact"),
    ("Coral Bleaching", "Ocean Ecosystem Disruption"),
]

# Mythological Figure Analysis Topics
mythological_figure_topics = [
    ("Zeus", "Greek Mythology"),
    ("Thor", "Norse Mythology"),
    ("Ra", "Egyptian Mythology"),
    ("Ganesha", "Hindu Mythology"),
    ("Amaterasu", "Japanese Mythology"),
    ("Quetzalcoatl", "Aztec Mythology"),
    ("Odin", "Norse Mythology"),
    ("Isis", "Egyptian Mythology"),
    ("Shiva", "Hindu Mythology"),
    ("Anansi", "African Mythology"),
    ("Aphrodite", "Greek Mythology"),
    ("Loki", "Norse Mythology"),
    ("Izanagi", "Japanese Mythology"),
    ("Coyote", "Native American Mythology"),
    ("Kali", "Hindu Mythology"),
]

# Python Algorithm Implementation Topics
python_algorithm_topics = [
    ("Bubble Sort", "Sorting"),
    ("Binary Search", "Searching"),
    ("Depth-First Search", "Graph Traversal"),
    ("Fibonacci Sequence", "Dynamic Programming"),
    ("Quicksort", "Sorting"),
    ("Dijkstra's Algorithm", "Shortest Path"),
    ("Merge Sort", "Sorting"),
    ("Breadth-First Search", "Graph Traversal"),
    ("Knapsack Problem", "Dynamic Programming"),
    ("Sieve of Eratosthenes", "Prime Numbers"),
    ("K-Means Clustering", "Machine Learning"),
    ("Huffman Coding", "Data Compression"),
    ("A* Search Algorithm", "Pathfinding"),
    ("Levenshtein Distance", "String Manipulation"),
    ("PageRank Algorithm", "Graph Analysis"),
]

# JavaScript DOM Manipulation Topics
js_dom_topics = [
    ("Create Element", "DOM Manipulation"),
    ("Event Listener", "User Interaction"),
    ("AJAX Request", "Asynchronous Operations"),
    ("Form Validation", "User Input"),
    ("DOM Traversal", "Element Selection"),
    ("Animation", "Visual Effects"),
    ("Local Storage", "Data Persistence"),
    ("Infinite Scroll", "Content Loading"),
    ("Drag and Drop", "User Interaction"),
    ("Modal Window", "UI Component"),
    ("Tabbed Interface", "Navigation"),
    ("Form Autocomplete", "User Input"),
    ("Image Slider", "Media Display"),
    ("Responsive Menu", "Navigation"),
    ("Dark Mode Toggle", "Theme Switching"),
]

# SQL Query Topics
sql_query_topics = [
    ("JOIN Operations", "Data Retrieval"),
    ("Subqueries", "Complex Queries"),
    ("Aggregate Functions", "Data Analysis"),
    ("Table Creation", "Database Structure"),
    ("Index Management", "Performance Optimization"),
    ("Stored Procedures", "Reusable Code"),
    ("Views", "Virtual Tables"),
    ("Transactions", "Data Integrity"),
    ("Triggers", "Automated Actions"),
    ("Window Functions", "Advanced Analysis"),
    ("Common Table Expressions", "Query Structuring"),
    ("PIVOT and UNPIVOT", "Data Transformation"),
    ("Full-Text Search", "Text Analysis"),
    ("Temporal Tables", "Historical Data"),
    ("JSON Operations", "Semi-Structured Data"),
]

# React Component Topics
react_component_topics = [
    ("Todo List", "State Management"),
    ("API Data Fetching", "External Data"),
    ("Form with Validation", "User Input"),
    ("Modal Dialog", "UI Interaction"),
    ("Responsive Navigation", "Layout"),
    ("Infinite Scroll", "Performance"),
    ("Authentication Flow", "User Management"),
    ("Data Visualization", "Charts and Graphs"),
    ("Drag and Drop Interface", "User Interaction"),
    ("Theme Switcher", "Customization"),
    ("Autocomplete Search", "User Input"),
    ("Image Gallery", "Media Display"),
    ("Progressive Form", "Multi-step Process"),
    ("Real-time Chat", "WebSocket Integration"),
    ("Lazy Loading", "Performance Optimization"),
]

# Data Structure Implementation Topics
data_structure_topics = [
    ("Linked List", "Linear Data Structure"),
    ("Binary Tree", "Hierarchical Structure"),
    ("Hash Table", "Associative Array"),
    ("Stack", "LIFO Structure"),
    ("Queue", "FIFO Structure"),
    ("Graph", "Network Structure"),
    ("Heap", "Priority Queue"),
    ("Trie", "Prefix Tree"),
    ("Circular Buffer", "Fixed-size Queue"),
    ("Bloom Filter", "Probabilistic Data Structure"),
    ("Skip List", "Probabilistic Data Structure"),
    ("Disjoint Set", "Union-Find Structure"),
    ("B-Tree", "Self-balancing Tree"),
    ("Red-Black Tree", "Self-balancing Tree"),
    ("LRU Cache", "Caching Mechanism"),
]

# Shell Scripting Topics
shell_scripting_topics = [
    ("File Backup", "System Administration"),
    ("Log Analysis", "Text Processing"),
    ("System Monitoring", "Performance Tracking"),
    ("Batch File Renaming", "File Management"),
    ("User Account Management", "System Administration"),
    ("Automated Deployment", "DevOps"),
    ("Network Diagnostics", "Troubleshooting"),
    ("Scheduled Tasks", "Automation"),
    ("Database Backup", "Data Management"),
    ("Web Scraping", "Data Collection"),
    ("Image Processing", "Batch Operations"),
    ("System Update", "Maintenance"),
    ("Log Rotation", "File Management"),
    ("SSH Key Management", "Security"),
    ("Docker Container Management", "Containerization"),
]

# Machine Learning Model Implementation Topics
ml_model_topics = [
    ("Linear Regression", "Supervised Learning"),
    ("K-Nearest Neighbors", "Classification"),
    ("Decision Tree", "Supervised Learning"),
    ("Random Forest", "Ensemble Learning"),
    ("Support Vector Machine", "Classification"),
    ("K-Means Clustering", "Unsupervised Learning"),
    ("Neural Network", "Deep Learning"),
    ("Naive Bayes", "Probabilistic Learning"),
    ("Logistic Regression", "Binary Classification"),
    ("Principal Component Analysis", "Dimensionality Reduction"),
    ("Gradient Boosting", "Ensemble Learning"),
    ("Long Short-Term Memory (LSTM)", "Sequence Learning"),
    ("Convolutional Neural Network", "Image Processing"),
    ("Reinforcement Learning Agent", "Decision Making"),
    ("Autoencoder", "Dimensionality Reduction"),
]

# Example of how to use these topics in prompts:
movie_review_prompt = "Write a movie review for '{}', directed by {}."
historical_event_prompt = "Summarize the key events and impacts of {} that occurred in {}."
tech_product_prompt = "Write a press release announcing the launch of the new {} by {}."
biographical_sketch_prompt = "Write a brief biographical sketch of {}, focusing on their contributions to {}."
environmental_impact_prompt = "Create an environmental impact report on the effects of {} in {}."
recipe_instruction_prompt = "Write a detailed recipe for {}, a popular {} dish."
travel_destination_prompt = "Create a travel guide for {}, located in {}, highlighting its main attractions and cultural significance."
scientific_explanation_prompt = "Explain the concept of {} in the field of {} for a general audience."
social_media_trend_prompt = "Analyze the rise and impact of {} as a trend in {} on social media platforms."
art_movement_prompt = "Analyze the key characteristics and influential artists of the {} movement, which emerged in {}."
psychological_disorder_prompt = "Explain the symptoms, causes, and treatment options for {}, a condition classified under {}."
musical_genre_prompt = "Describe the origins, key features, and notable artists of {} music, which originated in {}."
philosophical_concept_prompt = "Discuss the main ideas and implications of {} in the context of {}."
tech_innovation_prompt = "Explain the principles, applications, and potential impact of {} in the field of {}."
literary_genre_prompt = "Analyze the key characteristics and notable works of {} literature, which emerged in {}."
architectural_style_prompt = "Describe the main features and significant examples of {} architecture, prevalent during {}."
economic_theory_prompt = "Explain the core principles and impact of {}, developed by {}."
sports_event_prompt = "Summarize the highlights and significance of the {}, a major event in {}."
language_learning_prompt = "Create a beginner's guide to learning {}, a language from the family of {}."
climate_phenomenon_prompt = "Explain the causes and effects of {}, a climate phenomenon related to {}."
mythological_figure_prompt = "Describe the role and significance of {} in {}."
python_algorithm_prompt = "Implement the {} algorithm in Python, focusing on its application in {}."
js_dom_prompt = "Write JavaScript code to implement a {} functionality, demonstrating {}."
sql_query_prompt = "Write an SQL query to perform {}, showcasing its use in {}."
react_component_prompt = "Create a React component for a {}, emphasizing {}."
data_structure_prompt = "Implement a {} in your preferred programming language, explaining its use as a {}."
shell_scripting_prompt = "Write a shell script for {}, useful in {}."
ml_model_prompt = "Implement a {} model using a machine learning library, demonstrating its application in {}."

# Elementary Mathematics Prompts
elementary_math_prompts = [
    "If you have 15 apples and give 3 to your friend, then eat 2 yourself, how many apples do you have left?",
    "A rectangle has a length of 8 cm and a width of 5 cm. What is its area?",
    "What is the next number in the sequence: 2, 4, 6, 8, ...?",
    "If a pizza is cut into 8 equal slices and you eat 3 slices, what fraction of the pizza is left?",
    "A book costs $12. If you have $50, how many books can you buy, and how much money will you have left?",
]

# Algebra Prompts
algebra_prompts = [
    "Solve for x: 3x + 7 = 22",
    "If f(x) = 2x^2 + 3x - 5, what is f(2)?",
    "Simplify the expression: (3a^2b)(4ab^3)",
    "Factor the quadratic expression: x^2 + 5x + 6",
    "Solve the system of equations: y = 2x + 1 and y = -x + 7",
]

# Geometry Prompts
geometry_prompts = [
    "In a right triangle, one angle is 30°. What are the measures of the other two angles?",
    "A circle has a radius of 5 cm. What is its circumference? (Use π = 3.14)",
    "What is the volume of a rectangular prism with length 4 cm, width 3 cm, and height 5 cm?",
    "In an isosceles triangle, the base angles are equal. If one base angle is 50°, what is the measure of the vertex angle?",
    "The diagonal of a square is 10√2 cm. What is the area of the square?",
]

# Trigonometry Prompts
trigonometry_prompts = [
    "In a right triangle, the hypotenuse is 10 cm and one of the other sides is 6 cm. What is the length of the third side?",
    "Find the value of sin(30°) + cos(60°)",
    "A ladder 10 meters long leans against a vertical wall. If the angle between the ladder and the ground is 60°, how high up the wall does the ladder reach?",
    "Prove the identity: tan^2(θ) + 1 = sec^2(θ)",
    "Find all solutions to the equation: 2sin(x) = √3 in the interval [0, 2π]",
]

# Calculus Prompts
calculus_prompts = [
    "Find the derivative of f(x) = 3x^4 - 2x^2 + 5x - 1",
    "Evaluate the indefinite integral: ∫ (2x + 3)^2 dx",
    "Find the local maxima and minima of the function f(x) = x^3 - 3x^2 + 1",
    "Calculate the area under the curve y = x^2 from x = 0 to x = 2",
    "Find the volume of the solid formed by rotating the region bounded by y = x^2 and y = 2 about the y-axis",
]

# Statistics and Probability Prompts
stats_prob_prompts = [
    "A bag contains 5 red marbles, 3 blue marbles, and 2 green marbles. What is the probability of drawing a blue marble?",
    "The mean of a dataset is 15 and the standard deviation is 3. If a value is 2 standard deviations above the mean, what is its value?",
    "In a normal distribution, what percentage of the data falls within one standard deviation of the mean?",
    "A coin is flipped 5 times. What is the probability of getting exactly 3 heads?",
    "Calculate the correlation coefficient between two variables given the following data points: (1,2), (2,4), (3,5), (4,4), (5,5)",
]

# Number Theory Prompts
number_theory_prompts = [
    "Prove that the sum of two consecutive perfect squares is never a perfect square",
    "Find all prime numbers between 50 and 70",
    "Prove that for any integer n, n^3 - n is always divisible by 6",
    "What is the least common multiple (LCM) of 18, 24, and 36?",
    "Prove that √2 is irrational",
]

# Linear Algebra Prompts
linear_algebra_prompts = [
    "Find the eigenvalues and eigenvectors of the matrix [[3, 1], [1, 3]]",
    "Determine if the following set of vectors is linearly independent: {(1,2,3), (2,3,4), (3,5,7)}",
    "Solve the system of equations using Gaussian elimination: 2x + y - z = 1, x - y + 2z = 3, 3x + 2y - 3z = -2",
    "Find the rank of the matrix [[1, 2, 3], [2, 4, 6], [3, 6, 9]]",
    "Prove that the determinant of a 2x2 matrix [[a, b], [c, d]] is ad - bc",
]

# Mathematical Logic Prompts
math_logic_prompts = [
    "Prove by contradiction that there are infinitely many prime numbers",
    "Use truth tables to show that (p → q) is logically equivalent to (¬p ∨ q)",
    "Prove by induction that 1 + 2 + 3 + ... + n = n(n+1)/2 for all positive integers n",
    "Determine if the following argument is valid: All cats are mammals. Some mammals are not dogs. Therefore, some cats are not dogs.",
    "Prove that if n is an odd integer, then n^2 is odd",
]

# Applied Mathematics Prompts
applied_math_prompts = [
    "A population grows exponentially according to the formula P(t) = 1000e^(0.03t), where t is in years. How long will it take for the population to double?",
    "In a simple harmonic motion, the displacement is given by x(t) = 5 sin(2πt). Find the maximum velocity of the object.",
    "Using Newton's method, find an approximation of the square root of 7 to three decimal places",
    "A company's profit function is given by P(x) = -2x^2 + 120x - 1000, where x is the number of units produced. What production level maximizes profit?",
    "Solve the differential equation dy/dx = 2x + y with the initial condition y(0) = 1",
]

# Example of generating prompts:
movie_review_prompts = [movie_review_prompt.format(*topic) for topic in movie_review_topics]
historical_event_prompts = [historical_event_prompt.format(*topic) for topic in historical_event_topics]
tech_product_prompts = [tech_product_prompt.format(*topic) for topic in tech_product_topics]
biographical_sketch_prompts = [biographical_sketch_prompt.format(*topic) for topic in biographical_sketch_topics]
environmental_impact_prompts = [environmental_impact_prompt.format(*topic) for topic in environmental_impact_topics]
recipe_instruction_prompts = [recipe_instruction_prompt.format(*topic) for topic in recipe_instruction_topics]
travel_destination_prompts = [travel_destination_prompt.format(*topic) for topic in travel_destination_topics]
scientific_explanation_prompts = [scientific_explanation_prompt.format(*topic) for topic in
                                  scientific_explanation_topics]
social_media_trend_prompts = [social_media_trend_prompt.format(*topic) for topic in social_media_trend_topics]
art_movement_prompts = [art_movement_prompt.format(*topic) for topic in art_movement_topics]
psychological_disorder_prompts = [psychological_disorder_prompt.format(*topic) for topic in
                                  psychological_disorder_topics]
musical_genre_prompts = [musical_genre_prompt.format(*topic) for topic in musical_genre_topics]
philosophical_concept_prompts = [philosophical_concept_prompt.format(*topic) for topic in philosophical_concept_topics]
tech_innovation_prompts = [tech_innovation_prompt.format(*topic) for topic in tech_innovation_topics]
literary_genre_prompts = [literary_genre_prompt.format(*topic) for topic in literary_genre_topics]
architectural_style_prompts = [architectural_style_prompt.format(*topic) for topic in architectural_style_topics]
economic_theory_prompts = [economic_theory_prompt.format(*topic) for topic in economic_theory_topics]
sports_event_prompts = [sports_event_prompt.format(*topic) for topic in sports_event_topics]
language_learning_prompts = [language_learning_prompt.format(*topic) for topic in language_learning_topics]
climate_phenomenon_prompts = [climate_phenomenon_prompt.format(*topic) for topic in climate_phenomenon_topics]
mythological_figure_prompts = [mythological_figure_prompt.format(*topic) for topic in mythological_figure_topics]
python_algorithm_prompts = [python_algorithm_prompt.format(*topic) for topic in python_algorithm_topics]
js_dom_prompts = [js_dom_prompt.format(*topic) for topic in js_dom_topics]
sql_query_prompts = [sql_query_prompt.format(*topic) for topic in sql_query_topics]
react_component_prompts = [react_component_prompt.format(*topic) for topic in react_component_topics]
data_structure_prompts = [data_structure_prompt.format(*topic) for topic in data_structure_topics]
shell_scripting_prompts = [shell_scripting_prompt.format(*topic) for topic in shell_scripting_topics]
ml_model_prompts = [ml_model_prompt.format(*topic) for topic in ml_model_topics]

# Business and Economics Prompts
business_strategy_topics = [
    ("Blue Ocean Strategy", "Market Creation"),
    ("Disruptive Innovation", "Technology Adoption"),
    ("Porter's Five Forces", "Industry Analysis"),
    ("Balanced Scorecard", "Performance Measurement"),
    ("Lean Six Sigma", "Process Improvement"),
]

marketing_campaign_topics = [
    ("Viral Marketing", "Social Media"),
    ("Guerrilla Marketing", "Unconventional Tactics"),
    ("Influencer Marketing", "Social Proof"),
    ("Content Marketing", "Value-Added Information"),
    ("Email Marketing", "Direct Communication"),
]

financial_analysis_topics = [
    ("SWOT Analysis", "Strategic Planning"),
    ("ROI Calculation", "Investment Evaluation"),
    ("Break-Even Analysis", "Cost-Volume-Profit"),
    ("Cash Flow Forecasting", "Liquidity Management"),
    ("Ratio Analysis", "Financial Health Assessment"),
]

entrepreneurship_topics = [
    ("Lean Startup", "Minimum Viable Product"),
    ("Business Model Canvas", "Value Proposition"),
    ("Venture Capital", "Startup Funding"),
    ("Bootstrapping", "Self-Funding"),
    ("Pivot Strategy", "Business Model Adaptation"),
]

business_strategy_prompt = "Explain the concept of {} and its application in {}."
marketing_campaign_prompt = "Design a {} campaign focusing on {} for a new product launch."
financial_analysis_prompt = "Perform a {} to evaluate {} for a company in the tech industry."
entrepreneurship_prompt = "Discuss how {} can be applied in {} for a new startup."

# Health and Wellness Prompts
medical_condition_topics = [
    ("Type 2 Diabetes", "Metabolic Disorders"),
    ("Hypertension", "Cardiovascular Diseases"),
    ("Alzheimer's Disease", "Neurodegenerative Disorders"),
    ("Asthma", "Respiratory Conditions"),
    ("Osteoarthritis", "Musculoskeletal Disorders"),
]

fitness_routine_topics = [
    ("High-Intensity Interval Training", "Cardiovascular Fitness"),
    ("Yoga", "Flexibility and Mindfulness"),
    ("Strength Training", "Muscle Building"),
    ("Pilates", "Core Strength"),
    ("Calisthenics", "Bodyweight Exercises"),
]

nutrition_advice_topics = [
    ("Mediterranean Diet", "Heart Health"),
    ("Ketogenic Diet", "Weight Loss"),
    ("Plant-Based Diet", "Environmental Sustainability"),
    ("Intermittent Fasting", "Metabolic Health"),
    ("DASH Diet", "Hypertension Management"),
]

medical_condition_prompt = "Explain the causes, symptoms, and treatment options for {}, a condition classified under {}."
fitness_routine_prompt = "Design a {} workout plan aimed at improving {}."
nutrition_advice_prompt = "Provide a comprehensive guide to following a {} diet, emphasizing its benefits for {}."

# Education and Learning Prompts
study_technique_topics = [
    ("Pomodoro Technique", "Time Management"),
    ("Mind Mapping", "Visual Learning"),
    ("Spaced Repetition", "Long-term Retention"),
    ("Active Recall", "Information Retrieval"),
    ("Feynman Technique", "Concept Explanation"),
]

educational_theory_topics = [
    ("Constructivism", "Knowledge Construction"),
    ("Multiple Intelligences", "Learning Styles"),
    ("Bloom's Taxonomy", "Learning Objectives"),
    ("Flipped Classroom", "Student-Centered Learning"),
    ("Cooperative Learning", "Group Interaction"),
]

online_learning_resource_topics = [
    ("Massive Open Online Courses (MOOCs)", "Self-paced Learning"),
    ("Learning Management Systems", "Course Organization"),
    ("Educational Apps", "Mobile Learning"),
    ("Virtual Reality in Education", "Immersive Learning"),
    ("Gamification", "Engagement and Motivation"),
]

study_technique_prompt = "Explain how to effectively use the {} for {}, and provide examples of its application in different subjects."
educational_theory_prompt = "Discuss the principles of {} and how it impacts {} in modern education systems."
online_learning_prompt = "Evaluate the effectiveness of {} in promoting {} and suggest ways to optimize its use in online education."

# Law and Politics Prompts
legal_case_study_topics = [
    ("Brown v. Board of Education", "Civil Rights"),
    ("Roe v. Wade", "Reproductive Rights"),
    ("Citizens United v. FEC", "Campaign Finance"),
    ("Obergefell v. Hodges", "Same-Sex Marriage"),
    ("Miranda v. Arizona", "Criminal Procedure"),
]

political_system_topics = [
    ("Parliamentary Democracy", "Representative Government"),
    ("Federal Republic", "Power Distribution"),
    ("Constitutional Monarchy", "Traditional Leadership"),
    ("Direct Democracy", "Citizen Participation"),
    ("Single-Party State", "Centralized Authority"),
]

international_relations_topics = [
    ("Soft Power", "Cultural Influence"),
    ("Nuclear Deterrence", "International Security"),
    ("Economic Sanctions", "Diplomatic Pressure"),
    ("Humanitarian Intervention", "Global Responsibility"),
    ("Climate Change Diplomacy", "Environmental Cooperation"),
]

legal_case_study_prompt = "Analyze the key arguments and implications of {}, a landmark case in {}."
political_system_prompt = "Compare and contrast {} with other forms of {}, discussing its strengths and weaknesses."
international_relations_prompt = "Explain the concept of {} and its role in shaping {} in contemporary global politics."

# Generate prompts for each category
business_strategy_prompts = [business_strategy_prompt.format(*topic) for topic in business_strategy_topics]
marketing_campaign_prompts = [marketing_campaign_prompt.format(*topic) for topic in marketing_campaign_topics]
financial_analysis_prompts = [financial_analysis_prompt.format(*topic) for topic in financial_analysis_topics]
entrepreneurship_prompts = [entrepreneurship_prompt.format(*topic) for topic in entrepreneurship_topics]
medical_condition_prompts = [medical_condition_prompt.format(*topic) for topic in medical_condition_topics]
fitness_routine_prompts = [fitness_routine_prompt.format(*topic) for topic in fitness_routine_topics]
nutrition_advice_prompts = [nutrition_advice_prompt.format(*topic) for topic in nutrition_advice_topics]
study_technique_prompts = [study_technique_prompt.format(*topic) for topic in study_technique_topics]
educational_theory_prompts = [educational_theory_prompt.format(*topic) for topic in educational_theory_topics]
online_learning_prompts = [online_learning_prompt.format(*topic) for topic in online_learning_resource_topics]
legal_case_study_prompts = [legal_case_study_prompt.format(*topic) for topic in legal_case_study_topics]
political_system_prompts = [political_system_prompt.format(*topic) for topic in political_system_topics]
international_relations_prompts = [international_relations_prompt.format(*topic) for topic in
                                   international_relations_topics]

person = [
    "Ursula von der Leyen",
    "Xi Jinping",
    "Lula da Silva",
    "Volodymyr Zelenskyy",
    "Jacinda Ardern",
    "Cyril Ramaphosa",
    "Kamala Harris",
    "Rishi Sunak",
    "Olaf Scholz",
    "Narendra Modi",
    "Yoon Suk Yeol",
    "Anthony Albanese",
    "Mette Frederiksen",
    "Pedro Sánchez",
    "Recep Tayyip Erdoğan",
    "Gustavo Petro"
]

location = [
    "a climate change conference",
    "a G20 meeting",
    "a humanitarian aid event",
    "the World Economic Forum",
    "a UN General Assembly",
    "a regional trade summit",
    "an AI and ethics symposium",
    "a global health initiative launch",
    "a space exploration conference",
    "an international cultural festival",
    "a sustainable energy expo",
    "a cybersecurity summit"
]

topics = [
    "economic cooperation",
    "global security",
    "technological innovation",
    "public health initiatives",
    "climate action plans",
    "refugee crisis management",
    "artificial intelligence governance",
    "sustainable development goals",
    "nuclear non-proliferation",
    "digital currency regulations",
    "ocean conservation efforts",
    "space exploration collaboration",
    "human rights protection",
    "education in the digital age",
    "global food security",
    "renewable energy transition"
]

news_topics = [
    (person[i], person[j], location[k], topics[l])
    for i in range(len(person))
    for j in range(len(person))
    for k in range(len(location))
    for l in range(len(topics))
    if j > i
]

news_prompt = "Write a comprehensive news article about {}'s and {}'s joint appearance at {} focusing on their discussions about {}. Include relevant background information, potential outcomes, and expert opinions."

# News Article Prompts
news_prompts = {
    "news_articles": [news_prompt.format(*topic) for topic in news_topics]
}

# Arts and Culture Prompts
arts_culture_prompts = {
    "movie_reviews": movie_review_prompts,
    "art_movements": art_movement_prompts,
    "musical_genres": musical_genre_prompts,
    "literary_genres": literary_genre_prompts,
    "architectural_styles": architectural_style_prompts
}

# History and Society Prompts
history_society_prompts = {
    "historical_events": historical_event_prompts,
    "biographical_sketches": biographical_sketch_prompts,
    "social_media_trends": social_media_trend_prompts,
    "sports_events": sports_event_prompts,
    "mythological_figures": mythological_figure_prompts
}

# Science and Technology Prompts
science_tech_prompts = {
    "tech_products": tech_product_prompts,
    "scientific_explanations": scientific_explanation_prompts,
    "tech_innovations": tech_innovation_prompts,
    "climate_phenomena": climate_phenomenon_prompts
}

# Human Sciences Prompts
human_sciences_prompts = {
    "psychological_disorders": psychological_disorder_prompts,
    "philosophical_concepts": philosophical_concept_prompts,
    "economic_theories": economic_theory_prompts,
    "language_learning": language_learning_prompts
}

# Environment and Travel Prompts
environment_travel_prompts = {
    "environmental_impacts": environmental_impact_prompts,
    "travel_destinations": travel_destination_prompts
}

# Culinary Arts Prompts
culinary_arts_prompts = {
    "recipe_instructions": recipe_instruction_prompts
}

# Mathematics Prompts
math_prompts = {
    "Elementary Mathematics": elementary_math_prompts,
    "Algebra": algebra_prompts,
    "Geometry": geometry_prompts,
    "Trigonometry": trigonometry_prompts,
    "Calculus": calculus_prompts,
    "Statistics and Probability": stats_prob_prompts,
    "Number Theory": number_theory_prompts,
    "Linear Algebra": linear_algebra_prompts,
    "Mathematical Logic": math_logic_prompts,
    "Applied Mathematics": applied_math_prompts,
}

# Code Prompts
code_prompts = {
    "Python Algorithm Implementation": python_algorithm_prompts,
    "JavaScript DOM Manipulation": js_dom_prompts,
    "SQL Query Writing": sql_query_prompts,
    "React Component Development": react_component_prompts,
    "Data Structure Implementation": data_structure_prompts,
    "Shell Scripting": shell_scripting_prompts,
    "Machine Learning Model Implementation": ml_model_prompts,
}

# Business and Economics Prompts
business_economics_prompts = {
    "business_strategies": business_strategy_prompts,
    "marketing_campaigns": marketing_campaign_prompts,
    "financial_analysis": financial_analysis_prompts,
    "entrepreneurship": entrepreneurship_prompts
}

# Health and Wellness Prompts
health_wellness_prompts = {
    "medical_conditions": medical_condition_prompts,
    "fitness_routines": fitness_routine_prompts,
    "nutrition_advice": nutrition_advice_prompts
}

# Education and Learning Prompts
education_learning_prompts = {
    "study_techniques": study_technique_prompts,
    "educational_theories": educational_theory_prompts,
    "online_learning_resources": online_learning_prompts
}

# Law and Politics Prompts
law_politics_prompts = {
    "legal_case_studies": legal_case_study_prompts,
    "political_systems": political_system_prompts,
    "international_relations": international_relations_prompts
}

# All Prompt Groups
all_prompt_groups = {
    "Arts and Culture": arts_culture_prompts,
    "History and Society": history_society_prompts,
    "Science and Technology": science_tech_prompts,
    "Human Sciences": human_sciences_prompts,
    "Environment and Travel": environment_travel_prompts,
    "Culinary Arts": culinary_arts_prompts,
    "Mathematics": math_prompts,
    "Code": code_prompts,
    "Business and Economics": business_economics_prompts,
    "Health and Wellness": health_wellness_prompts,
    "Education and Learning": education_learning_prompts,
    "Law and Politics": law_politics_prompts,
    "News Articles": news_prompts
}

# System prompts for each subcategory
system_prompts = {
    "Arts and Culture": {
        "movie_reviews": "You are a professional film critic with extensive knowledge of cinema history, techniques, and genres. Provide insightful, balanced reviews that consider plot, acting, directing, cinematography, and overall impact.",
        "art_movements": "You are an art historian specializing in various art movements throughout history. Analyze artistic styles, key figures, and cultural contexts with academic rigor and clarity.",
        "musical_genres": "You are a musicologist with deep knowledge of various musical genres, their historical contexts, and technical aspects. Provide comprehensive analyses of musical styles, influences, and significant artists.",
        "literary_genres": "You are a literary scholar with expertise in various genres and periods of literature. Analyze literary works, movements, and authors with academic depth and cultural insight.",
        "architectural_styles": "You are an architectural historian with extensive knowledge of global architectural styles and their cultural significance. Provide detailed analyses of architectural features, historical contexts, and influential architects."
    },
    "History and Society": {
        "historical_events": "You are a historian specializing in global events across different time periods. Provide comprehensive, objective analyses of historical events, their causes, and their long-term impacts on society.",
        "biographical_sketches": "You are a biographer with expertise in crafting concise yet informative profiles of notable figures. Highlight key achievements, historical context, and lasting influence of the subject.",
        "social_media_trends": "You are a digital media analyst specializing in social media trends and their societal impacts. Provide insightful analyses of online phenomena, user behavior, and broader cultural implications.",
        "sports_events": "You are a sports journalist with extensive knowledge of various sports, their histories, and significant events. Provide comprehensive coverage of sporting events, including context, key moments, and broader significance.",
        "mythological_figures": "You are a mythologist with expertise in world mythologies and their cultural significance. Provide detailed explanations of mythological figures, their roles in various cultures, and their enduring influence."
    },
    "Science and Technology": {
        "tech_products": "You are a technology analyst with deep knowledge of various tech products and their market impacts. Provide insightful analyses of product features, market positioning, and potential societal effects.",
        "scientific_explanations": "You are a science communicator with expertise in translating complex scientific concepts for general audiences. Explain scientific phenomena clearly and engagingly, highlighting their real-world relevance.",
        "tech_innovations": "You are a technology futurist with a broad understanding of emerging technologies. Analyze innovative technologies, their potential applications, and their possible impacts on society and industry.",
        "climate_phenomena": "You are a climate scientist with expertise in global climate systems. Explain climate phenomena, their causes, and their impacts on the environment and human society with scientific accuracy and clarity."
    },
    "Human Sciences": {
        "psychological_disorders": "You are a clinical psychologist with expertise in various psychological disorders. Provide clear, sensitive explanations of mental health conditions, their symptoms, causes, and treatment options.",
        "philosophical_concepts": "You are a philosophy professor with broad knowledge of philosophical traditions and concepts. Explain complex philosophical ideas clearly, discussing their historical context and contemporary relevance.",
        "economic_theories": "You are an economist with expertise in various economic theories and their real-world applications. Analyze economic concepts, their historical development, and their impacts on policy and society.",
        "language_learning": "You are a linguist and language educator with expertise in language acquisition. Provide effective strategies and insights for learning new languages, considering linguistic principles and practical applications."
    },
    "Environment and Travel": {
        "environmental_impacts": "You are an environmental scientist specializing in human impacts on ecosystems. Provide comprehensive analyses of environmental issues, their causes, and potential solutions, backed by scientific data.",
        "travel_destinations": "You are a travel writer with extensive global experience. Create engaging, informative guides to various destinations, highlighting cultural attractions, practical tips, and unique experiences."
    },
    "Culinary Arts": {
        "recipe_instructions": "You are a professional chef with expertise in various cuisines and cooking techniques. Provide clear, detailed recipe instructions, including cultural context, ingredient information, and cooking tips."
    },
    "Mathematics": {
        "Elementary Mathematics": "You are a math educator specializing in elementary mathematics. Explain basic mathematical concepts clearly and engagingly, using relatable examples and step-by-step problem-solving approaches.",
        "Algebra": "You are a mathematics professor specializing in algebra. Explain algebraic concepts clearly, providing step-by-step solutions and real-world applications of abstract ideas.",
        "Geometry": "You are a geometry expert with a knack for spatial reasoning. Explain geometric principles clearly, using visual aids and real-world examples to illustrate concepts.",
        "Trigonometry": "You are a trigonometry specialist with expertise in both theoretical and applied aspects. Explain trigonometric concepts clearly, emphasizing their practical applications in various fields.",
        "Calculus": "You are a calculus professor with deep understanding of both differential and integral calculus. Explain complex calculus concepts clearly, emphasizing their applications in science and engineering.",
        "Statistics and Probability": "You are a statistician with expertise in data analysis and probability theory. Explain statistical concepts clearly, emphasizing their real-world applications and interpretation of results.",
        "Number Theory": "You are a number theorist with a passion for pure mathematics. Explain number theory concepts clearly, showcasing the beauty and depth of mathematical reasoning.",
        "Linear Algebra": "You are a linear algebra expert with knowledge of its applications in various fields. Explain linear algebra concepts clearly, emphasizing their importance in modern mathematics and computer science.",
        "Mathematical Logic": "You are a mathematical logician with expertise in formal systems and proof theory. Explain logical concepts clearly, showcasing the rigorous foundations of mathematical reasoning.",
        "Applied Mathematics": "You are an applied mathematician with experience in modeling real-world phenomena. Explain how mathematical concepts are applied to solve practical problems in various fields."
    },
    "Code": {
        "Python Algorithm Implementation": "You are a Python developer specializing in algorithm implementation. Provide clear, efficient, and well-commented Python code for various algorithms, explaining the logic and potential optimizations.",
        "JavaScript DOM Manipulation": "You are a front-end developer expert in JavaScript and DOM manipulation. Provide clear, efficient JavaScript code for various DOM operations, explaining best practices and potential pitfalls.",
        "SQL Query Writing": "You are a database administrator with expertise in SQL. Write clear, efficient SQL queries for various database operations, explaining the logic and potential optimizations.",
        "React Component Development": "You are a React developer with expertise in component-based architecture. Develop clear, reusable React components, explaining key concepts and best practices in React development.",
        "Data Structure Implementation": "You are a computer scientist specializing in data structures. Implement various data structures clearly and efficiently, explaining their properties, use cases, and potential trade-offs.",
        "Shell Scripting": "You are a systems administrator expert in shell scripting. Write clear, efficient shell scripts for various system operations, explaining the commands and potential portability issues.",
        "Machine Learning Model Implementation": "You are a machine learning engineer with expertise in various ML algorithms. Implement machine learning models clearly and efficiently, explaining the underlying principles and potential applications."
    },
    "Business and Economics": {
        "business_strategies": "You are a business strategist with expertise in various industries and market dynamics. Analyze business strategies comprehensively, considering market trends, competitive landscapes, and potential outcomes.",
        "marketing_campaigns": "You are a marketing expert with experience in various marketing channels and strategies. Design effective marketing campaigns, explaining target audience considerations, messaging strategies, and success metrics.",
        "financial_analysis": "You are a financial analyst with expertise in various financial instruments and market dynamics. Conduct thorough financial analyses, explaining key metrics, market trends, and potential investment strategies.",
        "entrepreneurship": "You are a successful entrepreneur and business consultant. Provide insightful advice on starting and growing businesses, covering aspects such as idea validation, funding, and scaling strategies."
    },
    "Health and Wellness": {
        "medical_conditions": "You are a medical professional with broad knowledge of various health conditions. Explain medical conditions clearly and accurately, covering symptoms, causes, diagnostic procedures, and treatment options.",
        "fitness_routines": "You are a certified fitness trainer with expertise in various exercise modalities. Design effective fitness routines, explaining proper form, progression strategies, and considerations for different fitness levels.",
        "nutrition_advice": "You are a registered dietitian with expertise in various dietary approaches. Provide evidence-based nutrition advice, explaining the science behind different diets and their potential health impacts."
    },
    "Education and Learning": {
        "study_techniques": "You are an education psychologist specializing in learning strategies. Explain effective study techniques, their scientific basis, and how to apply them to different subjects and learning styles.",
        "educational_theories": "You are an education theorist with expertise in various pedagogical approaches. Analyze educational theories comprehensively, discussing their historical context, key principles, and practical applications.",
        "online_learning_resources": "You are an e-learning specialist with expertise in digital education tools and platforms. Evaluate online learning resources, discussing their features, effectiveness, and best practices for implementation."
    },
    "Law and Politics": {
        "legal_case_studies": "You are a legal scholar with expertise in various areas of law. Analyze legal cases comprehensively, explaining key arguments, legal principles, and broader implications for jurisprudence.",
        "political_systems": "You are a political scientist specializing in comparative politics. Analyze different political systems, their historical development, key features, and impacts on governance and society.",
        "international_relations": "You are an international relations expert with deep understanding of global politics. Analyze international relations concepts and events, considering historical context, power dynamics, and potential global impacts."
    },
    "News Articles": {
        "news_articles": "You are an experienced journalist with expertise in global current events. Write comprehensive, compelling, even if totally invented news articles, providing context, analysis, and diverse perspectives on complex issues."
    }
}


# Function to convert prompt lists to prompt-system pairs
def convert_to_prompt_system_pairs(prompt_groups, system_prompts):
    prompt_system_pairs = {}
    for category, subcategories in prompt_groups.items():
        prompt_system_pairs[category] = {}
        for subcategory, prompts in subcategories.items():
            prompt_system_pairs[category][subcategory] = [
                (prompt, system_prompts[category][subcategory])
                for prompt in prompts
            ]
    return prompt_system_pairs


# Convert prompt lists to prompt-system pairs
all_prompt_groups = convert_to_prompt_system_pairs(all_prompt_groups, system_prompts)

# Count how many prompts are there overall (minus news)
total_prompts = sum(
    [len(subcategory) for category in all_prompt_groups for subcategory in all_prompt_groups[category] if
     category != "News Articles"])

# Make all_prompt_groups contain only a few of news articles
all_prompt_groups["News Articles"] = {
    "news_articles": random.sample(all_prompt_groups["News Articles"]["news_articles"], total_prompts // 2)
}

all_prompts = []
for group in all_prompt_groups.values():
    for prompts in group.values():
        all_prompts.extend(prompts)


# Function to get a random subset of prompts
def get_random_prompts(num_prompts=20):
    random_state = random.getstate()
    random.seed()
    result = random.sample(all_prompts, num_prompts)
    random.setstate(random_state)
    return result


if __name__ == "__main__":
    print("\n".join(["\t".join(x) for x in get_random_prompts()]))
