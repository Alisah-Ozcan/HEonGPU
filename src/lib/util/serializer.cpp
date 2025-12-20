#include <heongpu/util/serializer.h>

namespace heongpu
{
    namespace serializer
    {

        std::vector<uint8_t> to_buffer(const std::stringstream& ss)
        {
            const std::string& str = ss.str();
            return {str.begin(), str.end()};
        }

        void from_buffer(std::stringstream& ss,
                         const std::vector<uint8_t>& buffer)
        {
            ss.str(std::string(buffer.begin(), buffer.end()));
        }

        std::vector<uint8_t> compress(const std::vector<uint8_t>& data)
        {
            uLongf bound = compressBound(data.size());
            std::vector<uint8_t> out(bound);

            if (::compress(out.data(), &bound, data.data(), data.size()) !=
                Z_OK)
            {
                throw std::runtime_error("Zlib compression failed");
            }

            out.resize(bound);
            return out;
        }

        std::vector<uint8_t> decompress(const std::vector<uint8_t>& data)
        {
            // Estimate output size (may need adjustment for very large data)
            std::vector<uint8_t> out(data.size() * 4);
            uLongf outSize = out.size();

            if (::uncompress(out.data(), &outSize, data.data(), data.size()) !=
                Z_OK)
            {
                throw std::runtime_error("Zlib decompression failed");
            }

            out.resize(outSize);
            return out;
        }

    } // namespace serializer
} // namespace heongpu