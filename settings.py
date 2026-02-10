# from win32com.demos.excelRTDServer import EXCEL_TLB_LCID, EXCEL_TLB_GUID

MB = 1024 * 1024
DEFAULT_FILE_SIZE_LIMIT = 10 * MB
SUPPORTED_FILE_GROUPS = {

    # ─────────────── Office文档类 ───────────────
    ('.docx', '.doc', '.wps', '.dot', '.dotx', '.rtf', '.odt', '.pages'): {
        'description': '文档文件',
        'size_limit': 30 * MB,
        'category': 'document',
    },

    # ─────────────── 表格类 ───────────────
    ('.xlsx', '.xls', '.xlsm', '.xlt', '.xltx', '.csv', '.tsv', '.et', '.ett', '.ods', '.numbers'): {
        'description': '表格文件',
        'size_limit': 5 * MB,
        'category': 'spreadsheet',
    },

    # ─────────────── 演示文稿类 ───────────────
    ('.pptx', '.ppt', '.pptm', '.pot', '.potx', '.dps', '.dpt', '.odp', '.key', '.keynote'): {
        'description': '演示文稿',
        'size_limit': 40 * MB,
        'category': 'presentation',
    },

    # ─────────────── PDF/电子书类 ───────────────
    ('.pdf', '.ofd', '.epub', '.mobi', '.azw', '.azw3', '.fb2', '.chm'): {
        'description': '电子文档',
        'size_limit': 50 * MB,
        'category': 'ebook',
    },
    # ─────────────── 纯文本类 ───────────────
    ('.txt', '.log', '.ini', '.cfg', '.conf', '.properties', '.toml', '.env', '.gitignore', '.dockerignore'): {
        'description': '文本文件',
        'size_limit': 15 * MB,
        'category': 'text',
    },

    # ─────────────── 网页类 ───────────────
    ('.html', '.htm', '.xhtml', '.xml', '.json', '.yaml', '.yml', '.css', '.scss', '.less', '.sass'): {
        'description': '网页相关文件',
        'size_limit': 20 * MB,
        'category': 'web',
    },

    # ─────────────── 图像类 ───────────────
    ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif', '.ico', '.heic', '.heif', '.psd', '.ai', '.eps'): {
        'description': '图像文件',
        'size_limit': 50 * MB,
        'category': 'image',
    },

    # ─────────────── 矢量图形类 ───────────────
    ('.svg', '.odg', '.dwg', '.dxf'): {
        'description': '矢量图形文件',
        'size_limit': 30 * MB,
        'category': 'vector',
    },

    # ─────────────── 标记语言类 ───────────────
    ('.md', '.markdown', '.mdown', '.mkd', '.rst', '.tex', '.latex'): {
        'description': '标记文档',
        'size_limit': 10 * MB,
        'category': 'markup',
    },

    # ─────────────── 编程源码类 ───────────────
    ('.py', '.java', '.cpp', '.c', '.h', '.hpp', '.cs', '.js', '.ts', '.jsx', '.tsx', '.php', '.rb', '.go',
     '.rs', '.swift', '.kt', '.scala', '.dart', '.lua', '.pl', '.pm', '.sh', '.bash', '.zsh', '.fish',
     '.ps1', '.bat', '.cmd', '.vbs', '.sql', '.r', '.m', '.mat', '.f', '.for', '.f90', '.f95','.sql'): {
        'description': '源代码文件',
        'size_limit': 10 * MB,
        'category': 'code',
    },


    # # ─────────────── 字体文件类 ───────────────
    # ('.ttf', '.otf', '.woff', '.woff2', '.eot', '.fon'): {
    #     'description': '字体文件',
    #     'size_limit': 30 * MB,
    #     'category': 'font',
    # },

    # ─────────────── 音频类 ───────────────
    ('.mp3', '.wav', '.flac', '.aac', '.ogg', '.oga', '.m4a', '.wma', '.ape', '.mid', '.midi'): {
        'description': '音频文件',
        'size_limit': 100 * MB,
        'category': 'audio',
    },

    # ─────────────── 视频类 ───────────────
    ('.mp4', '.avi', '.mov', '.wmv', '.flv', '.mkv', '.webm', '.m4v', '.mpg', '.mpeg', '.rm', '.rmvb', '.3gp', '.ogv'): {
        'description': '视频文件',
        'size_limit': 200 * MB,
        'category': 'video',
    },

    # # ─────────────── 压缩/归档类 ───────────────
    # ('.zip', '.rar', '.7z', '.tar', '.gz', '.bz2', '.xz', '.z', '.lz', '.lzma', '.cab', '.arj', '.zst'): {
    #     'description': '压缩文件',
    #     'size_limit': 200 * MB,
    #     'category': 'archive',
    # },
    #
    # # ─────────────── 数据库类 ───────────────
    # ('.db', '.sqlite', '.sqlite3', '.mdb', '.accdb', '.dbf', '.mdf', '.ndf', '.ldf', '.frm', '.myd', '.myi'): {
    #     'description': '数据库文件',
    #     'size_limit': 100 * MB,
    #     'category': 'database',
    # },
    #
    # # ─────────────── 3D/模型类 ───────────────
    # ('.stl', '.obj', '.fbx', '.3ds', '.max', '.blend', '.ma', '.mb', '.dae', '.ply', '.gltf', '.glb'): {
    #     'description': '3D模型文件',
    #     'size_limit': 100 * MB,
    #     'category': '3d',
    # },
    #
    # # ─────────────── 虚拟化/磁盘类 ───────────────
    # ('.iso', '.vhd', '.vhdx', '.vmdk', '.ova', '.ovf', '.qcow2', '.raw', '.img', '.bin', '.cue'): {
    #     'description': '磁盘镜像文件',
    #     'size_limit': 1024 * MB,  # 1GB
    #     'category': 'disk_image',
    # },
    #
    # # ─────────────── 可执行文件类 ───────────────
    # ('.exe', '.msi', '.dmg', '.pkg', '.deb', '.rpm', '.apk', '.ipa', '.app', '.jar', '.war', '.ear'): {
    #     'description': '可执行文件',
    #     'size_limit': 200 * MB,
    #     'category': 'executable',
    # },

    # # ─────────────── 系统/配置文件 ───────────────
    # ('.reg', '.inf', '.sys', '.dll', '.so', '.dylib', '.a', '.lib'): {
    #     'description': '系统文件',
    #     'size_limit': 50 * MB,
    #     'category': 'system',
    # },
}

# 如果需要保持向后兼容，可以生成平铺的配置
SUPPORTED_FILE_SIZE_LIMIT = {}
SUPPORTED_FILE_EXTENDS = {}

for extensions, config in SUPPORTED_FILE_GROUPS.items():
    for ext in extensions:
        # 复制配置避免共享引用
        SUPPORTED_FILE_SIZE_LIMIT[ext] = config['size_limit']
        SUPPORTED_FILE_EXTENDS[ext] = config['description']

# 分片配置
MAX_CHUNK_SIZE = 500 #文档拆分的最大字符数
CHUNK_OVERLAP = 0    #文档拆分时的交叉字符数
MIN_CHUNK_SIZE = 50  #文档拆分的最小字符数

VECTOR_BATCH_SIZE = 4 # 向量化批次的数量
#因为向量化非常慢，如果文档过大，则需要长时间加载，所以只做部分向量化，保证每个文件向量化的时间不超过1分钟。
MAX_VECTORS_SIZE_PER_FILE = 16*20 #预估CPU环境，每秒可以加载16个500字符串长度的文本，限定时间为20秒。
PARSE_DOC_TIME_OUT = 60
PIPLINE_DOC_TIME_OUT = 60
MIN_SCORE = 0.6

EXCEL_MAX_ROWS=6000
EXCEL_MAX_COLUMNS=128
PDF_MAX_PAGES=100
PDF_MAX_OCR_PAGES=30
