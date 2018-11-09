# NBA-PBP-Data

This is a project I started to analyze NBA play-by-play data using Python. My goal was to be able to parse an array of play-by-play data for a given matchup between two teams and build an adversarial synergy graph -- the approach was inspired by the paper below. While I found the learning process challenging, I was able to employ the basic approach, ableit in a more "creative" fashion. 

The Adversarial Synergy Graph script takes a CSV of lineup scores -- i.e. the points scored and allowed for two "adversarial" five-person basketball teams -- and "guesses" the ideal graph structure that corresponds with the spread in the score. If there's a transitive property that explains the difference in score, manipulating a series of unweighted graphs can provide insight about agent-by-agent performance, leading to better insight about "optimal" team formation. While I'm new to Python, I've enjoyed the challenge and looking forward to updating this project throughout the 2018-2019 season.

Here's the research paper that prompted me to attempt to write this script:

http://ala2015.csc.liv.ac.uk/papers/ala15_liemhetcharat.pdf


Any feedback is welcome!
