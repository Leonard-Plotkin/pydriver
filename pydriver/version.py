# -*- coding: utf-8 -*-
__version_info__ = {
	'major': 0,
	'minor': 1,
	'micro': 0,
	'release': '',
}
__version_info__['short'] = '{0[major]}.{0[minor]}'.format(__version_info__)
__version_info__['full'] = '{0[major]}.{0[minor]}.{0[micro]}{0[release]}'.format(__version_info__)
__version__ = __version_info__['full']
