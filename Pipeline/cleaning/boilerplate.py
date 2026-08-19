from collections import Counter

from cv2 import line


SAFE_REMOVE = [

    "©",
    "copyright",
    "all rights reserved",

    "contact us",
    "privacy",
    "privacy policy",
    "terms",
    "terms of use",
    "terms of service",

    "accessibility",
    "accessibility statement",

    "top of page",
    "back to top",

    "work for us",
    "careers",
    "vacancies",
    "jobs",
    "recruitment",

    "cqc logo",

    "connect with us",
    "follow us",
    "follow us on",

    "site by",
    "website by",
    "web design",
    "powered by",

    "find us around the web",

    "important links",

    "newsletter",
    "sign up for our newsletter",

    "donate",
    "donate now",
    "make a donation",

    "press office",

    "maps and car parking",
    "accessable guides",

    "cookie policy",
    "cookie settings",
    "sitemap",
]


CONTEXT_REMOVE = [

    # Contact / Organization
    "contact information",
    "contact details",
    "get in touch",
    "our location",
    "find our location",
    "directions",

    "office hours",
    "opening hours",
    "clinic hours",
    "business hours",
    "hours & info",
    "hours and info",


    # Communication
    "email",
    "phone",
    "telephone",
    "tel",
    "fax",
    "address",
    "street",
    "suite",
    "ste",


    # Navigation
    "home",
    "menu",
    "navigation",
    "main menu",

    "search",
    "search this site",
    "site search",

    "menu toggle",
    "close menu",


    # Page navigation
    "skip to content",
    "skip to main content",

    "breadcrumb",
    "breadcrumbs",
    "you are here",


    # Social
    "share",
    "share this page",
    "follow us",
    "connect with us",


    # Account
    "sign in",
    "login",
    "log in",
    "register",
    "my account",
    "patient portal",


    # Cookies
    "cookie",
    "cookies",
    "cookie settings",
    "accept cookies",
    "manage cookies",
    "reject cookies",


    # Feedback
    "feedback",
    "give us feedback",
    "report a problem",
    "help us improve",


    # Time
    "monday",
    "friday",
    "pacific time",


    # Links
    "related links",
    "important links",
    "patient information links",
    "patient information resources",
]



def is_safe_remove(line: str):

    text = line.lower().strip()

    return any(
        item.lower() in text
        for item in SAFE_REMOVE
    )



def is_context_remove(line: str):

    text = line.lower().strip()

    return any(
        text == item.lower()
        for item in CONTEXT_REMOVE
    )



def remove_footer_block(text: str):

    lines = text.split("\n")

    footer_start_keywords = [

        "hours & info",
        "hours and info",

        "useful information",
        "important links",

        "find us around the web",

        "cqc overall rating",
        "cqc logo",

        "connect with us",
        "follow us",

        "milton keynes university hospital",
        "standing way",

        "patient portal",

        "newsletter",
        "subscribe",

        "contact us",
        "copyright",
        "site by",

        "©",
        "copyright",
        "all rights reserved",
    
        "contact us",
        "privacy",
        "privacy policy",
        "terms",
        "terms of use",
        "terms of service",
    
        "accessibility",
        "accessibility statement",
    
        "top of page",
        "back to top",
    
        "work for us",
        "careers",
        "vacancies",
        "jobs",
        "recruitment",
    
        "cqc logo",
    
        "connect with us",
        "follow us",
        "follow us on",
    
        "site by",
        "website by",
        "web design",
        "powered by",
    
        "find us around the web",
    
        "important links",
    
        "newsletter",
        "sign up for our newsletter",
    
        "donate",
        "donate now",
        "make a donation",
    
        "press office",
    
        "maps and car parking",
        "accessable guides",
    
        "cookie policy",
        "cookie settings",
        "sitemap",

        "share this:",
        "related",
        "find a sages member in your area",
    ]


    cut_index = None


    # ندور في آخر 40% فقط
    start_search = int(len(lines) * 0.6)


    for i in range(start_search, len(lines)):

        line = lines[i].lower().strip()



        for keyword in footer_start_keywords:

            if line == keyword:
                cut_index = i
                break

        if cut_index:
            break


    if cut_index:

        return "\n".join(
            lines[:cut_index]
        )


    return text



def find_repeated_lines(text: str, min_count=3):

    lines = [
        line.strip()
        for line in text.split("\n")
        if line.strip()
    ]

    counter = Counter(lines)


    return {
        line
        for line, count in counter.items()
        if count >= min_count
    }



def remove_repeated_lines(text, repeated_lines):

    return "\n".join(
        line
        for line in text.split("\n")
        if line.strip() not in repeated_lines
    )